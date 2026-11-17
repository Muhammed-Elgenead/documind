import json
import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.llm import LLMError, get_llm
from app.agents.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT
from app.db.models import Chunk, Document, DocumentStatus, Message
from app.db.session import get_session
from app.main import app
from app.services.embeddings import HashEmbedder
from app.services.retrieval import RetrievalMode, RetrievalOptions, retrieve
from tests.conftest import TEST_USER_ID

EMB = HashEmbedder()
VECTOR_K8 = RetrievalOptions(mode=RetrievalMode.VECTOR, k=8)


class FakeLLM:
    """Scripted LLM: yields `tokens`, or raises `error` after yielding them."""

    def __init__(self, tokens: list[str], error: Exception | None = None):
        self.tokens = tokens
        self.error = error
        self.calls: list[tuple[str, list[dict]]] = []

    async def stream_answer(self, system, messages):
        self.calls.append((system, messages))
        for t in self.tokens:
            yield t
        if self.error:
            raise self.error


def use_llm(llm: FakeLLM) -> FakeLLM:
    app.dependency_overrides[get_llm] = lambda: llm
    return llm


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def seed(
    session_factory, docs: dict[str, list[tuple[int, str]]], status="ready", user_id=TEST_USER_ID
):
    """docs: filename -> [(page, text), ...]. Returns {filename: document_id}."""
    ids = {}
    async with session_factory() as s:
        for filename, pages in docs.items():
            doc = Document(filename=filename, status=status, page_count=len(pages), user_id=user_id)
            s.add(doc)
            await s.flush()
            for i, (page, text) in enumerate(pages):
                s.add(
                    Chunk(
                        document_id=doc.id,
                        page=page,
                        chunk_index=i,
                        text=text,
                        embedding=EMB.embed_query(text),
                    )
                )
            ids[filename] = doc.id
        await s.commit()
    return ids


CORPUS = {
    "finance.pdf": [
        (3, "Quarterly revenue grew twelve percent driven by cloud sales"),
        (7, "Operating costs rose because of new office leases"),
    ],
    "hr.docx": [(2, "The company hired forty engineers this year")],
}


# ---------- No database needed ----------


async def test_blank_question_rejected(client):
    session = AsyncMock(spec=AsyncSession)

    async def _get_session():
        yield session

    app.dependency_overrides[get_session] = _get_session
    for q in ["", "   "]:
        resp = await client.post("/chat", json={"question": q})
        assert resp.status_code == 422
    session.add.assert_not_called()


# ---------- Retrieval ----------


async def test_retrieve_ranks_by_similarity_with_document_and_page(session_factory):
    await seed(session_factory, CORPUS)
    async with session_factory() as s:
        results = await retrieve(
            s, EMB, "how much did quarterly revenue grow", VECTOR_K8, user_id=None
        )
    assert len(results) == 3
    top = results[0]
    assert (top.filename, top.page) == ("finance.pdf", 3)
    assert "revenue" in top.text
    assert results[0].score >= results[1].score >= results[2].score


async def test_retrieve_respects_k_status_and_document_filter(session_factory):
    ids = await seed(session_factory, CORPUS)
    await seed(session_factory, {"pending.pdf": [(1, "revenue revenue revenue")]}, "processing")
    async with session_factory() as s:
        assert len(await retrieve(s, EMB, "revenue", RetrievalOptions(k=2), user_id=None)) == 2
        all_files = {r.filename for r in await retrieve(s, EMB, "revenue", VECTOR_K8, user_id=None)}
        only_hr = await retrieve(
            s, EMB, "revenue", VECTOR_K8, document_ids=[ids["hr.docx"]], user_id=None
        )
    assert "pending.pdf" not in all_files
    assert {r.filename for r in only_hr} == {"hr.docx"}


# ---------- POST /chat ----------


async def test_chat_streams_answer_and_saves_turn_with_citations(db_client, session_factory):
    await seed(session_factory, CORPUS)
    llm = use_llm(FakeLLM(["Revenue grew ", "12% [1]", ". Bogus [9]."]))

    resp = await db_client.post("/chat", json={"question": "How much did revenue grow?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)

    names = [e for e, _ in events]
    assert names == ["start", "token", "token", "token", "done"]
    start, done = events[0][1], events[-1][1]
    assert start["sources"] == 3
    assert "".join(d["text"] for e, d in events if e == "token") == done["answer"]
    assert done["answer"] == "Revenue grew 12% [1]. Bogus [9]."

    # [1] maps to the top-ranked source; the out-of-range [9] is dropped.
    assert done["citations"] == [
        {
            "number": 1,
            "document_id": done["citations"][0]["document_id"],
            "filename": "finance.pdf",
            "page": 3,
            "chunk_id": done["citations"][0]["chunk_id"],
            "text": CORPUS["finance.pdf"][0][1],
        }
    ]

    # Prompt: fixed system prompt, numbered sources and the question in one user turn.
    (system, messages) = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert len(messages) == 1
    assert '<source number="1" document="finance.pdf" page="3">' in messages[0]["content"]
    assert messages[0]["content"].endswith("Question: How much did revenue grow?")

    async with session_factory() as s:
        rows = (await s.execute(select(Message).order_by(Message.id))).scalars().all()
    assert [(m.role, m.content) for m in rows] == [
        ("user", "How much did revenue grow?"),
        ("assistant", done["answer"]),
    ]
    assert rows[0].citations is None
    assert rows[1].citations == done["citations"]
    assert rows[1].id == done["message_id"]
    assert str(rows[1].conversation_id) == done["conversation_id"] == start["conversation_id"]


async def test_follow_up_uses_history_and_same_conversation(db_client, session_factory):
    await seed(session_factory, CORPUS)
    use_llm(FakeLLM(["Revenue grew 12% [1]."]))
    first = parse_sse((await db_client.post("/chat", json={"question": "Revenue?"})).text)
    conv_id = first[-1][1]["conversation_id"]

    llm = use_llm(FakeLLM(["Forty engineers [2]."]))
    resp = await db_client.post(
        "/chat", json={"question": "And hiring?", "conversation_id": conv_id}
    )
    events = parse_sse(resp.text)
    assert events[-1][1]["conversation_id"] == conv_id

    (_, messages) = llm.calls[0]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "Revenue?"
    assert messages[1]["content"] == "Revenue grew 12%."  # old [1] marker stripped

    conv = (await db_client.get(f"/conversations/{conv_id}")).json()
    assert conv["title"] == "Revenue?"
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["messages"][3]["citations"][0]["number"] == 2


async def test_no_documents_answers_not_found_without_llm(db_client):
    llm = use_llm(FakeLLM(["should not be used"]))
    events = parse_sse((await db_client.post("/chat", json={"question": "Anything?"})).text)
    assert events[0] == (
        "start",
        {"conversation_id": events[0][1]["conversation_id"], "sources": 0},
    )
    assert events[-1][0] == "done"
    assert events[-1][1]["answer"] == NOT_FOUND_MESSAGE
    assert events[-1][1]["citations"] == []
    assert llm.calls == []


async def test_model_not_found_answer_has_no_citations(db_client, session_factory):
    await seed(session_factory, CORPUS)
    use_llm(FakeLLM([NOT_FOUND_MESSAGE]))
    events = parse_sse((await db_client.post("/chat", json={"question": "Weather?"})).text)
    assert events[-1][1]["answer"] == NOT_FOUND_MESSAGE
    assert events[-1][1]["citations"] == []


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (LLMError("The LLM is rate limited. Please try again shortly."), "rate limited"),
        (RuntimeError("boom"), "Something went wrong"),
    ],
)
async def test_llm_failure_emits_error_event(db_client, session_factory, error, message):
    await seed(session_factory, CORPUS)
    use_llm(FakeLLM(["Partial"], error=error))
    events = parse_sse((await db_client.post("/chat", json={"question": "Revenue?"})).text)
    assert [e for e, _ in events] == ["start", "token", "error"]
    assert message in events[-1][1]["message"]
    async with session_factory() as s:
        roles = (await s.execute(select(Message.role))).scalars().all()
    assert roles == ["user"]  # the question is kept; no broken answer is saved


async def test_unknown_conversation_404(db_client):
    use_llm(FakeLLM(["x"]))
    resp = await db_client.post(
        "/chat", json={"question": "hi", "conversation_id": str(uuid.uuid4())}
    )
    assert resp.status_code == 404
    assert (await db_client.get(f"/conversations/{uuid.uuid4()}")).status_code == 404


async def test_history_is_trimmed_to_whole_turns(session_factory):
    from app.services import chat as chat_service

    async with session_factory() as s:
        conv = await chat_service.create_conversation(s, TEST_USER_ID, "q1")
        for role, text in [
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
            ("user", "dangling"),
        ]:
            await chat_service.add_message(s, conv.id, role, text)
        # limit 4 -> last four are a1,q2,a2,dangling -> drop leading assistant and trailing user
        assert await chat_service.recent_history(s, conv.id, 4) == [
            ("user", "q2"),
            ("assistant", "a2"),
        ]
        assert await chat_service.recent_history(s, conv.id, 0) == []


async def test_document_status_constant_matches():
    assert DocumentStatus.READY == "ready"


async def test_list_conversations_newest_first(db_client, session_factory):
    await seed(session_factory, CORPUS)
    use_llm(FakeLLM(["ok [1]."]))
    for q in ["First question", "Second question"]:
        await db_client.post("/chat", json={"question": q})
    convs = (await db_client.get("/conversations")).json()
    assert [c["title"] for c in convs] == ["Second question", "First question"]
    assert set(convs[0]) == {"id", "title", "created_at"}
    assert len((await db_client.get("/conversations?limit=1")).json()) == 1

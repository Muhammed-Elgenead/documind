import uuid

from app.agents.citations import build_citations, cited_numbers
from app.agents.prompts import (
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    build_messages,
    format_sources,
    strip_citation_markers,
)
from app.services.retrieval import RetrievedChunk


def _chunk(i: int, filename: str = "report.pdf", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=100 + i,
        document_id=uuid.UUID(int=i),
        filename=filename,
        page=page,
        chunk_index=i,
        text=f"text of chunk {i}",
        score=0.9,
    )


def test_system_prompt_has_the_rules():
    assert NOT_FOUND_MESSAGE in SYSTEM_PROMPT
    assert "[2]" in SYSTEM_PROMPT
    assert "only" in SYSTEM_PROMPT.lower()


def test_sources_are_numbered_with_document_and_page():
    out = format_sources([_chunk(1, "a.pdf", 3), _chunk(2, 'we"ird<.pdf', 7)])
    assert '<source number="1" document="a.pdf" page="3">\ntext of chunk 1\n</source>' in out
    assert 'number="2" document="we&quot;ird&lt;.pdf" page="7"' in out
    assert out.startswith("<sources>") and out.endswith("</sources>")


def test_build_messages_puts_sources_only_in_last_turn_and_strips_old_markers():
    history = [("user", "first q"), ("assistant", "Answer one [1][2] and more [3].")]
    msgs = build_messages(history, "second q", [_chunk(1)])
    assert msgs[0] == {"role": "user", "content": "first q"}
    assert msgs[1] == {"role": "assistant", "content": "Answer one and more."}
    assert msgs[2]["role"] == "user"
    assert "<sources>" in msgs[2]["content"]
    assert msgs[2]["content"].endswith("Question: second q")


def test_strip_citation_markers_handles_lists():
    assert strip_citation_markers("A [1, 2] b [3].") == "A b."


def test_cited_numbers_order_dedupe_and_range():
    answer = "X [2]. Y [1][2]. Z [3, 9]. W [0]. V [8]."
    assert cited_numbers(answer, source_count=8) == [2, 1, 3, 8]


def test_cited_numbers_none():
    assert cited_numbers(NOT_FOUND_MESSAGE, 8) == []


def test_build_citations_maps_numbers_to_sources():
    chunks = [_chunk(1, "a.pdf", 4), _chunk(2, "b.pdf", 9)]
    cites = build_citations("Claim [2]. Other [1]. Bogus [5].", chunks)
    assert cites == [
        {
            "number": 2,
            "document_id": str(uuid.UUID(int=2)),
            "filename": "b.pdf",
            "page": 9,
            "chunk_id": 102,
            "text": "text of chunk 2",
        },
        {
            "number": 1,
            "document_id": str(uuid.UUID(int=1)),
            "filename": "a.pdf",
            "page": 4,
            "chunk_id": 101,
            "text": "text of chunk 1",
        },
    ]

import pytest

from app.services.chunking import CHARS_PER_TOKEN, chunk_pages, estimate_tokens
from app.services.extraction import PageText


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_empty_input_gives_no_chunks():
    assert chunk_pages([]) == []
    assert chunk_pages([PageText(1, ""), PageText(2, "   ")]) == []


def test_short_page_is_one_chunk_with_its_page_number():
    chunks = chunk_pages([PageText(3, "Hello world.")])
    assert len(chunks) == 1
    assert chunks[0].page == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "Hello world."


def test_long_page_splits_into_chunks_within_size():
    text = _words(2000)
    chunks = chunk_pages([PageText(1, text)], chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(estimate_tokens(c.text) <= 100 for c in chunks)
    # Chunks are reasonably full (not tiny slivers), except possibly the last one.
    assert all(estimate_tokens(c.text) >= 80 for c in chunks[:-1])


def test_consecutive_chunks_overlap():
    chunks = chunk_pages([PageText(1, _words(2000))], chunk_size=100, overlap=20)
    for a, b in zip(chunks, chunks[1:], strict=False):
        a_words, b_words = a.text.split(), b.text.split()
        shared = [w for w in b_words if w in set(a_words)]
        # The shared words are exactly the end of `a` and the start of `b`...
        assert shared
        assert b_words[: len(shared)] == shared == a_words[-len(shared) :]
        # ...and roughly the requested overlap, never more.
        assert 15 <= estimate_tokens(" ".join(shared)) <= 20


def test_zero_overlap_has_no_shared_words():
    chunks = chunk_pages([PageText(1, _words(1000))], chunk_size=50, overlap=0)
    seen: set[str] = set()
    for c in chunks:
        words = set(c.text.split())
        assert not (words & seen)
        seen |= words


def test_all_words_are_covered_in_order():
    text = _words(1500)
    chunks = chunk_pages([PageText(1, text)], chunk_size=80, overlap=15)
    rebuilt: list[str] = []
    for c in chunks:
        for w in c.text.split():
            if not rebuilt or int(w[1:]) > int(rebuilt[-1][1:]):
                rebuilt.append(w)
    assert rebuilt == text.split()


def test_chunks_never_cross_pages_and_indexes_are_global():
    pages = [PageText(1, _words(600, "a")), PageText(2, ""), PageText(3, _words(600, "c"))]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 3}
    for c in chunks:
        prefix = "a" if c.page == 1 else "c"
        assert all(w.startswith(prefix) for w in c.text.split())
    # Page order is preserved.
    pages_in_order = [c.page for c in chunks]
    assert pages_in_order == sorted(pages_in_order)


def test_oversized_word_is_hard_split():
    giant = "x" * (100 * CHARS_PER_TOKEN * 3)  # 3x the chunk size, no spaces
    chunks = chunk_pages([PageText(1, giant)], chunk_size=100, overlap=10)
    assert len(chunks) >= 3
    assert all(estimate_tokens(c.text) <= 100 for c in chunks)
    assert "".join(c.text for c in chunks).count("x") >= len(giant)


def test_default_size_is_about_500_tokens():
    chunks = chunk_pages([PageText(1, _words(5000))])
    assert all(estimate_tokens(c.text) <= 500 for c in chunks)
    assert estimate_tokens(chunks[0].text) > 450


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "text")], chunk_size=size, overlap=overlap)

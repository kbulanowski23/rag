"""Reasoning traces must never reach the user, and ordinary answers must pass
through byte-for-byte. The hard case is streaming: the tag arrives split across
token boundaries, which is exactly what a regex over the finished text cannot
help with, because there is no finished text yet when the UI renders."""

from __future__ import annotations

import pytest

from rag_core.rag.reasoning import ReasoningStripper, strip_reasoning


def stream(chunks: list[str]) -> str:
    s = ReasoningStripper()
    return "".join(s.feed(c) for c in chunks) + s.flush()


# -- whole-text ---------------------------------------------------------------

def test_removes_the_reasoning_block():
    text = "<think>Let me check the sources. Source 2 says seven years.</think>\n\nSeven years [2]."
    assert strip_reasoning(text) == "Seven years [2]."


def test_answer_without_reasoning_is_untouched():
    assert strip_reasoning("Seven years [1].") == "Seven years [1]."


def test_unterminated_block_yields_nothing():
    # Truncated mid-thought: there is no answer to show, and showing the partial
    # trace would be worse than showing nothing.
    assert strip_reasoning("<think>I should start by") == ""


# -- streaming ----------------------------------------------------------------

def test_tag_split_across_chunks():
    # The case this whole module exists for.
    assert stream(["<th", "ink>hid", "den</thi", "nk>", "Visible [1]."]) == "Visible [1]."


def test_token_by_token_matches_whole_text():
    full = "<think>weighing the options</think>\n\nSeven years [2]."
    assert stream(list(full)) == strip_reasoning(full)


def test_text_resembling_a_tag_is_not_swallowed():
    # A "<" that never becomes <think> must be released, not held forever.
    assert stream(["The value is <", "100 and >", "50."]) == "The value is <100 and >50."


def test_partial_tag_at_end_of_stream_is_released():
    assert stream(["Answer <thin"]) == "Answer <thin"


def test_leading_blank_lines_after_the_block_are_trimmed():
    assert stream(["<think>x</think>", "\n\n", "Answer."]) == "Answer."


def test_content_before_the_block_survives():
    assert stream(["Intro. ", "<think>x</think>", " Rest."]) == "Intro.  Rest."


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13])
def test_chunking_never_changes_the_result(size: int):
    full = "<think>reasoning [9] here</think>\n\nThe answer is seven years [2][3]."
    chunks = [full[i:i + size] for i in range(0, len(full), size)]
    assert stream(chunks) == "The answer is seven years [2][3]."


def test_custom_tags():
    s = ReasoningStripper("<reasoning>", "</reasoning>")
    assert s.feed("<reasoning>hm</reasoning>Answer.") + s.flush() == "Answer."

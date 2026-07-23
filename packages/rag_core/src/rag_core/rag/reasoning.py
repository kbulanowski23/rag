"""Strip reasoning traces out of a model's answer.

Reasoning models (qwen3, deepseek-r1, gpt-oss and friends) emit their chain of
thought inline, wrapped in `<think>...</think>`, before the actual answer. Shown
to a user it is worse than noise: it is longer than the answer, it contradicts
itself as it works things out, and it leaks the prompt's structure. On one
measured qwen3 answer it was 1319 of 2644 characters -- half the response.

This cannot be done with a regex over the finished text, because the text is
streamed: the tag itself is routinely split across token boundaries, arriving as
`<th` + `ink>`, and the UI renders each token as it lands. So it is a small state
machine that holds back only what might turn out to be part of a tag.

Removing the trace does not stop the model generating it -- the tokens are still
produced and still cost latency. Suppressing generation is a per-model prompt or
API concern (some gateways expose a `reasoning_effort` or `chat_template_kwargs`
switch); this is only about what reaches the user.
"""

from __future__ import annotations


def _prefix_overlap(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `tag`.

    This is what makes the streaming case correct: if a chunk ends in `<thi`,
    that might be the start of `<think>` or might be literal text, and we cannot
    know until more arrives. Hold back exactly that much and no more.
    """
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class ReasoningStripper:
    """Feed it streamed text; it returns only what belongs in the answer."""

    def __init__(self, open_tag: str = "<think>", close_tag: str = "</think>") -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        self._buf = ""
        self._inside = False
        # Reasoning blocks are followed by blank lines. Once they are gone the
        # answer would start with stray whitespace, so trim until real content.
        self._seen_output = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []

        while self._buf:
            if self._inside:
                idx = self._buf.find(self.close_tag)
                if idx == -1:
                    # Everything so far is reasoning. Keep only what could be
                    # the leading part of the closing tag.
                    keep = _prefix_overlap(self._buf, self.close_tag)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[idx + len(self.close_tag):]
                self._inside = False
                continue

            idx = self._buf.find(self.open_tag)
            if idx == -1:
                keep = _prefix_overlap(self._buf, self.open_tag)
                emit = self._buf[: len(self._buf) - keep] if keep else self._buf
                self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                if emit:
                    out.append(emit)
                break

            if idx:
                out.append(self._buf[:idx])
            self._buf = self._buf[idx + len(self.open_tag):]
            self._inside = True

        return self._clean("".join(out))

    def flush(self) -> str:
        """Emit whatever is still held back at end of stream.

        An unterminated `<think>` means the model was cut off mid-thought; there
        is no answer to salvage, so the remainder is dropped rather than shown.
        """
        if self._inside:
            self._buf = ""
            return ""
        rest, self._buf = self._buf, ""
        return self._clean(rest)

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        if not self._seen_output:
            text = text.lstrip()
            if not text:
                return ""
            self._seen_output = True
        return text


def strip_reasoning(text: str, open_tag: str = "<think>", close_tag: str = "</think>") -> str:
    """Non-streaming convenience wrapper, for the /chat (non-SSE) path."""
    s = ReasoningStripper(open_tag, close_tag)
    return (s.feed(text) + s.flush()).strip()

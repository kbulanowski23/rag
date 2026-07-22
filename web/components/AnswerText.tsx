"use client";

import type { Source } from "@/lib/types";

/**
 * Renders an answer, turning the model's [n] markers into clickable chips.
 *
 * Rendering the answer as plain text with citation chips, rather than as
 * markdown, is a deliberate choice: it removes a markdown parser and sanitiser
 * from the dependency list, and it means model output can never inject markup.
 * If rich formatting becomes necessary, add a sanitising renderer -- never
 * dangerouslySetInnerHTML on model output.
 */
export default function AnswerText({
  text,
  sources,
  onCiteClick,
}: {
  text: string;
  sources: Source[];
  onCiteClick?: (index: number) => void;
}) {
  const parts: React.ReactNode[] = [];
  const pattern = /\[(\d{1,2})\]/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = pattern.exec(text)) !== null) {
    const n = Number(match[1]);
    // Only linkify numbers that map to a source we actually have. A stray "[5]"
    // in quoted document text must stay literal rather than become a dead link.
    if (n < 1 || n > sources.length) continue;

    if (match.index > last) parts.push(text.slice(last, match.index));
    parts.push(
      <button
        key={`c${key++}`}
        className="cite"
        title={sources[n - 1]?.title || `Source ${n}`}
        onClick={() => onCiteClick?.(n)}
      >
        {n}
      </button>,
    );
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));

  return <div className="msg-content">{parts}</div>;
}

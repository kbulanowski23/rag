"use client";

import { useState } from "react";
import type { Source } from "@/lib/types";

function SourceCard({
  source,
  cited,
  highlighted,
}: {
  source: Source;
  cited: boolean;
  highlighted: boolean;
}) {
  const [open, setOpen] = useState(false);
  const pages =
    source.page_start > 0
      ? source.page_end && source.page_end !== source.page_start
        ? `pp. ${source.page_start}–${source.page_end}`
        : `p. ${source.page_start}`
      : "";

  // Which retriever found this, and at what rank. Surfacing it is what makes
  // relevance debuggable without opening a terminal.
  const provenance = Object.entries(source.retrievers)
    .map(([name, rank]) => `${name}#${rank}`)
    .join(" · ");

  return (
    <div
      id={`source-${source.index}`}
      className={`source${cited ? "" : " uncited"}${highlighted ? " highlight" : ""}`}
    >
      <div className="source-head">
        <span className="source-num">{source.index}</span>
        <span className="source-title">{source.title || source.filename}</span>
      </div>
      <div className="source-meta">
        {[pages, provenance, source.score ? `score ${source.score.toFixed(4)}` : "",
          source.extraction_source === "ocr" ? "OCR" : ""]
          .filter(Boolean)
          .join("  ·  ")}
      </div>
      <div className={`source-text${open ? " open" : ""}`}>{source.text}</div>
      {source.text.length > 220 && (
        <button className="source-toggle" onClick={() => setOpen((v) => !v)}>
          {open ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}

export default function SourcePanel({
  sources,
  cited,
  highlighted,
}: {
  sources: Source[];
  cited: number[];
  highlighted: number | null;
}) {
  return (
    <aside className="sidebar">
      <h2>Sources {sources.length > 0 && `(${sources.length})`}</h2>
      {sources.length === 0 ? (
        <p className="source-meta">
          Retrieved documents will appear here, with the page reference for each
          excerpt.
        </p>
      ) : (
        sources.map((s) => (
          <SourceCard
            key={s.chunk_id}
            source={s}
            // Before the answer completes, `cited` is empty; show everything at
            // full opacity rather than dimming the whole list mid-stream.
            cited={cited.length === 0 || cited.includes(s.index)}
            highlighted={highlighted === s.index}
          />
        ))
      )}
    </aside>
  );
}

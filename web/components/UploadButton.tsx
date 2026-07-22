"use client";

import { useRef, useState } from "react";
import { uploadDocument } from "@/lib/api";

/**
 * Single-file upload for demos and spot checks. Bulk loading goes through the
 * worker CLI -- a browser upload holds an API pod for the whole extraction,
 * which for a large scanned PDF can be minutes.
 */
export default function UploadButton({ onUploaded }: { onUploaded?: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  const handle = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setStatus(`Indexing ${file.name}…`);
    try {
      const result = await uploadDocument(file);
      setStatus(
        result.chunks_indexed > 0
          ? `${file.name}: ${result.chunks_indexed} chunks${result.ocr_pages ? `, ${result.ocr_pages} OCR pages` : ""}`
          : `${file.name}: no text extracted`,
      );
      onUploaded?.();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
      window.setTimeout(() => setStatus(""), 6000);
    }
  };

  return (
    <span className="upload">
      <button className="btn" onClick={() => inputRef.current?.click()} disabled={busy}>
        {busy ? "Indexing…" : "Upload"}
      </button>
      <input
        ref={inputRef}
        type="file"
        onChange={handle}
        accept=".pdf,.docx,.doc,.pptx,.xlsx,.txt,.md,.html,.csv,.png,.jpg,.jpeg,.tif,.tiff"
      />
      {status && <span className="upload-status">{status}</span>}
    </span>
  );
}

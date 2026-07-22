"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AnswerText from "./AnswerText";
import SourcePanel from "./SourcePanel";
import UploadButton from "./UploadButton";
import { getConfig, getIndexStats, streamChat } from "@/lib/api";
import type { ChatTurn, EffectiveConfig, IndexStats, Message, Source } from "@/lib/types";

const EXAMPLES = [
  "What is the retention period for claims documentation?",
  "Summarise the approval steps in the procurement policy.",
  "Which documents mention the incident response escalation path?",
];

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [config, setConfig] = useState<EffectiveConfig | null>(null);
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Shown in the header so it is always obvious which model and index a given
    // environment is actually using -- the first thing anyone asks when results
    // differ between home and work.
    getConfig().then(setConfig).catch(() => {});
    getIndexStats().then(setStats).catch(() => {});
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const last = messages[messages.length - 1];
  const activeSources: Source[] = last?.sources ?? [];
  const activeCited: number[] = last?.cited ?? [];

  const send = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || busy) return;

      const history: ChatTurn[] = messages
        .filter((m) => !m.error)
        .slice(-6)
        .map((m) => ({ role: m.role, content: m.content }));

      const userMsg: Message = { id: `u${Date.now()}`, role: "user", content: text };
      const assistantId = `a${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        userMsg,
        { id: assistantId, role: "assistant", content: "", streaming: true, sources: [], cited: [] },
      ]);
      setInput("");
      setBusy(true);

      const controller = new AbortController();
      abortRef.current = controller;

      const patch = (fn: (m: Message) => Message) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

      try {
        await streamChat(text, {
          history,
          signal: controller.signal,
          onEvent: (event) => {
            switch (event.type) {
              case "sources":
                patch((m) => ({ ...m, sources: event.sources }));
                break;
              case "token":
                // Appending per token keeps the UI honest about generation speed.
                patch((m) => ({ ...m, content: m.content + event.text }));
                break;
              case "done":
                patch((m) => ({
                  ...m,
                  streaming: false,
                  cited: event.cited,
                  timings: event.timings_ms,
                }));
                break;
              case "error":
                patch((m) => ({ ...m, streaming: false, error: event.message }));
                break;
            }
          },
        });
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        if (!controller.signal.aborted) {
          patch((m) => ({ ...m, streaming: false, error: message }));
        }
      } finally {
        patch((m) => ({ ...m, streaming: false }));
        setBusy(false);
        abortRef.current = null;
      }
    },
    [busy, messages],
  );

  const stop = () => abortRef.current?.abort();

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends, Shift+Enter inserts a newline.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send(input);
    }
  };

  const jumpToSource = (index: number) => {
    setHighlighted(index);
    document.getElementById(`source-${index}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => setHighlighted(null), 1600);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Enterprise RAG Search</h1>
        <div className="spacer" />
        {stats?.exists && (
          <span className="badge">
            {stats.documents ?? 0} docs · {stats.chunks ?? 0} chunks
          </span>
        )}
        {config && (
          <span className="badge" title={`${config.llm_provider} @ ${config.llm_base_url}`}>
            {config.llm_model}
          </span>
        )}
        {config && <span className="badge">{config.fusion}</span>}
        <UploadButton onUploaded={() => getIndexStats().then(setStats).catch(() => {})} />
      </header>

      <div className="body">
        <main className="main">
          <div className="messages" ref={scrollRef}>
            <div className="messages-inner">
              {messages.length === 0 && (
                <div className="empty">
                  <h2>Ask a question about your documents</h2>
                  <p>Answers are generated only from indexed content, with page-level citations.</p>
                  <div className="examples">
                    {EXAMPLES.map((q) => (
                      <button key={q} className="example" onClick={() => void send(q)}>
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((m) => (
                <div key={m.id} className={`msg msg-${m.role}`}>
                  <div className="msg-role">{m.role === "user" ? "You" : "Assistant"}</div>
                  {m.role === "user" ? (
                    <div className="msg-content">{m.content}</div>
                  ) : (
                    <>
                      <AnswerText
                        text={m.content}
                        sources={m.sources ?? []}
                        onCiteClick={jumpToSource}
                      />
                      {m.streaming && <span className="cursor" />}
                      {m.error && <div className="msg-error">⚠ {m.error}</div>}
                      {m.timings && (
                        <div className="timings">
                          retrieval {m.timings.retrieval}ms · generation {m.timings.generation}ms
                        </div>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="composer">
            <div className="composer-inner">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Ask a question…"
                rows={1}
                disabled={busy}
              />
              {busy ? (
                <button className="btn" onClick={stop}>
                  Stop
                </button>
              ) : (
                <button className="btn btn-primary" onClick={() => void send(input)} disabled={!input.trim()}>
                  Send
                </button>
              )}
            </div>
            <div className="hint">Enter to send · Shift+Enter for a new line</div>
          </div>
        </main>

        <SourcePanel sources={activeSources} cited={activeCited} highlighted={highlighted} />
      </div>
    </div>
  );
}

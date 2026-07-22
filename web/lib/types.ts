export interface Source {
  index: number;
  chunk_id: string;
  doc_id: string;
  title: string;
  filename: string;
  source_uri: string;
  page_start: number;
  page_end: number;
  score: number;
  extraction_source: string;
  retrievers: Record<string, number>;
  text: string;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  cited?: number[];
  streaming?: boolean;
  error?: string;
  timings?: Record<string, number>;
}

export interface EffectiveConfig {
  env: string;
  llm_provider: string;
  llm_model: string;
  llm_base_url: string;
  embedding_provider: string;
  embedding_dim: number;
  index: string;
  fusion: string;
  final_k: number;
  rerank_enabled: boolean;
  ocr_enabled: boolean;
}

export interface IndexStats {
  index: string;
  exists: boolean;
  chunks?: number;
  documents?: number;
}

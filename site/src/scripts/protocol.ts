// Messages between the demo UI and the inference worker.

export interface Alt {
  text: string;
  prob: number;
}

export interface TokenEvent {
  id: number;
  text: string; // decoded text this token added (may be "" mid-codepoint)
  prob: number;
  top: Alt[];
}

export type ToWorker =
  | { type: "load"; url: string; wasmBase: string }
  | {
      type: "generate";
      prompt: string;
      temperature: number;
      topP: number;
      maxTokens: number;
      seed: number;
    }
  | { type: "stop" };

export type FromWorker =
  | { type: "progress"; loaded: number; total: number; cached: boolean }
  | { type: "ready"; backend: string; params: number; seqLen: number; loadMs: number }
  | { type: "prompt"; tokens: number; ms: number }
  | { type: "tokens"; items: TokenEvent[]; position: number; generated: number; elapsedMs: number }
  | { type: "done"; reason: "eos" | "length" | "context" | "stopped"; generated: number; elapsedMs: number }
  | { type: "error"; message: string };

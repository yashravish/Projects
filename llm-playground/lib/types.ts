export type Run = {
  id: string;
  model: string;
  prompt: string;
  params: string;
  tokens_input: number;
  tokens_output: number;
  latency_ms: number;
  cost_usd: number;
  created_at: string;
};

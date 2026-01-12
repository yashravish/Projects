import { z } from 'zod';

export const querySchema = z.object({
  question: z.string().min(3).max(1000),
});

export type QueryInput = z.infer<typeof querySchema>;

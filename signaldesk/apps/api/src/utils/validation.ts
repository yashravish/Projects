import { z } from 'zod';

/**
 * Schema for validating CUID path parameters (Prisma default)
 */
export const uuidParamSchema = z.object({
  id: z.string().cuid({ message: 'Invalid ID format. Expected a valid CUID.' }),
});

/**
 * Schema for validating collection ID path parameters
 */
export const collectionIdParamSchema = z.object({
  collectionId: z.string().cuid({ message: 'Invalid collection ID format. Expected a valid CUID.' }),
});

/**
 * Schema for validating collection with document ID path parameters
 */
export const collectionDocumentParamSchema = z.object({
  collectionId: z.string().cuid({ message: 'Invalid collection ID format. Expected a valid CUID.' }),
  id: z.string().cuid({ message: 'Invalid document ID format. Expected a valid CUID.' }),
});

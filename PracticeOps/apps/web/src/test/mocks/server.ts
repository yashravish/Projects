/**
 * MSW Server Setup
 *
 * Mock Service Worker server for API mocking in tests.
 */

import { setupServer } from "msw/node";
import { handlers } from "./handlers";

export const server = setupServer(...handlers);


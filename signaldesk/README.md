# SignalDesk Lite

A full-stack AI-powered document intelligence platform that enables users to upload documents, index them into a vector store with semantic embeddings, and query them using RAG (Retrieval-Augmented Generation).

![SignalDesk Lite](https://img.shields.io/badge/Stack-TypeScript%20%7C%20Next.js%20%7C%20Fastify%20%7C%20Prisma-blue)

## Features

- **Document Upload & Processing**: Upload PDF, TXT, and MD files (up to 10MB)
- **Semantic Indexing**: Automatic text chunking and embedding generation with OpenAI
- **RAG Query System**: Natural language queries with grounded answers and inline citations
- **Collections**: Organize documents into projects
- **Analytics Dashboard**: Track usage, token consumption, and estimated costs
- **Background Processing**: BullMQ-based job queue for document processing
- **Multi-tenant**: User authentication with JWT and isolated data access
- **Distinctive Design**: Modern UI with Playfair Display and IBM Plex Sans fonts

## Tech Stack

### Frontend
- **Next.js 14** (App Router)
- **React 18** with TypeScript
- **TailwindCSS 3** + shadcn/ui components
- **TanStack Query v5** (React Query)
- **Framer Motion** for animations

### Backend
- **Fastify 4** with TypeScript
- **Prisma 5** ORM
- **PostgreSQL** via Neon (serverless)
- **pgvector** for semantic search
- **BullMQ 5** job queue
- **Upstash Redis** (serverless)

### AI/ML
- **OpenAI API** (text-embedding-3-small, gpt-4o-mini)
- **RAG** (Retrieval-Augmented Generation)

## Prerequisites

- **Node.js** 20+
- **pnpm** 8+
- **Neon PostgreSQL** account (free tier available)
- **Upstash Redis** account (free tier available)
- **OpenAI API key** (optional - app works in stub mode without it)

## Cloud Setup

### 1. Neon PostgreSQL

1. Create account at [neon.tech](https://neon.tech)
2. Create a new project
3. Enable pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
4. Copy both connection strings:
   - **DATABASE_URL**: Pooled connection for app queries
   - **DIRECT_URL**: Direct connection for Prisma migrations

### 2. Upstash Redis

1. Create account at [upstash.com](https://upstash.com)
2. Create a Redis database
3. Copy the connection details:
   - **REDIS_URL**: Redis connection string
   - **UPSTASH_REDIS_REST_URL**: REST API URL
   - **UPSTASH_REDIS_REST_TOKEN**: REST API token

## Installation

1. **Clone the repository**
   ```bash
   cd "Service Desk Lite"
   ```

2. **Install dependencies**
   ```bash
   pnpm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your cloud credentials:
   ```env
   # Neon PostgreSQL
   DATABASE_URL="postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require"
   DIRECT_URL="postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require"

   # Upstash Redis
   REDIS_URL="rediss://default:xxx@xxx.upstash.io:6379"
   UPSTASH_REDIS_REST_URL="https://xxx.upstash.io"
   UPSTASH_REDIS_REST_TOKEN="your-token"

   # JWT Secret (generate a secure random string)
   JWT_SECRET="your-256-bit-secret-change-in-production"

   # OpenAI API Key (optional)
   OPENAI_API_KEY="sk-your-openai-api-key"

   # API Configuration
   API_PORT=3001
   API_HOST=0.0.0.0

   # Frontend
   NEXT_PUBLIC_API_URL=http://localhost:3001

   NODE_ENV=development
   ```

4. **Build shared package**
   ```bash
   pnpm --filter @signaldesk/shared build
   ```

5. **Generate Prisma client**
   ```bash
   pnpm db:generate
   ```

6. **Run database migrations**
   ```bash
   pnpm db:migrate
   ```

   This will create all tables and the pgvector index.

7. **Optional: Seed test data**
   ```bash
   pnpm db:seed
   ```

   This creates a demo user:
   - Email: `demo@signaldesk.com`
   - Password: `password123`

## Development

Start both the API server and web app:

```bash
pnpm dev
```

This runs:
- API server on `http://localhost:3001`
- Web app on `http://localhost:3000`

### Individual Commands

```bash
# Start API only
pnpm --filter api dev

# Start Web only
pnpm --filter web dev

# Run tests
pnpm test

# Run API tests
pnpm test:api

# Lint code
pnpm lint

# Format code
pnpm format

# Open Prisma Studio (database GUI)
pnpm db:studio
```

## Architecture

```
signaldesk-lite/
├── apps/
│   ├── api/                    # Fastify backend
│   │   ├── src/
│   │   │   ├── routes/         # API endpoints
│   │   │   ├── services/       # Business logic
│   │   │   ├── jobs/           # BullMQ workers
│   │   │   ├── middleware/     # Auth middleware
│   │   │   └── index.ts        # Server entry
│   │   ├── prisma/
│   │   │   ├── schema.prisma   # Database schema
│   │   │   └── migrations/     # SQL migrations
│   │   └── storage/            # Uploaded files
│   └── web/                    # Next.js frontend
│       └── src/
│           ├── app/            # App Router pages
│           ├── components/     # React components
│           └── lib/            # Utilities
└── packages/
    └── shared/                 # Shared types & schemas
        └── src/
            ├── constants.ts    # Configuration
            └── schemas/        # Zod validation
```

## API Endpoints

### Authentication
- `POST /v1/auth/signup` - Register new user
- `POST /v1/auth/login` - Login and get JWT cookie
- `POST /v1/auth/logout` - Logout and clear cookie
- `GET /v1/auth/me` - Get current user

### Collections
- `GET /v1/collections` - List user's collections
- `POST /v1/collections` - Create collection
- `GET /v1/collections/:id` - Get collection details
- `PATCH /v1/collections/:id` - Update collection
- `DELETE /v1/collections/:id` - Delete collection

### Documents
- `GET /v1/collections/:id/documents` - List documents
- `POST /v1/collections/:id/documents` - Upload document
- `GET /v1/documents/:id` - Get document with chunks
- `DELETE /v1/documents/:id` - Delete document

### Query (RAG)
- `POST /v1/collections/:id/query` - Query documents with AI

### Analytics
- `GET /v1/analytics` - Global analytics
- `GET /v1/collections/:id/analytics` - Collection analytics

## RAG Query Flow

1. **Embed Question**: Convert user question to 1536-dim vector
2. **Retrieve Chunks**: Find top-6 similar chunks via pgvector cosine similarity
3. **Build Context**: Format retrieved chunks as context
4. **Generate Answer**: Call GPT-4o-mini with context
5. **Extract Citations**: Parse inline citations and sources
6. **Log Query**: Store query, answer, tokens, and latency

## Stub Mode

Without an OpenAI API key, the app runs in stub mode:
- Documents are still chunked and stored
- Queries return the first K chunks (no similarity search)
- Stub responses are generated instead of AI answers

## Testing

```bash
# Run all tests
pnpm test

# Run API tests only
pnpm test:api

# Run web tests only
pnpm test:web
```

Tests include:
- **Unit tests**: Chunking, auth service, utilities
- **Integration tests**: API routes (auth, collections, documents, query)

## Deployment

### API (Railway, Render, Fly.io)

1. Set environment variables in hosting platform
2. Build command: `pnpm install && pnpm build`
3. Start command: `pnpm --filter api start`
4. Port: `3001` (or `$PORT`)

### Web (Vercel, Netlify)

1. Set `NEXT_PUBLIC_API_URL` to deployed API URL
2. Build command: `pnpm install && pnpm --filter web build`
3. Output directory: `apps/web/.next`

## Design System

### Typography
- **Display**: Playfair Display (serif)
- **Body**: IBM Plex Sans (sans-serif)

### Colors
- **Background**: Dark navy (`#0A0D14`)
- **Primary**: Blue (`#3B82F6`)
- **Accent**: Purple (`#8B5CF6`)
- **Text**: Off-white (`#FAFAFA`)

### Animations
- Page transitions with Framer Motion
- Card hover effects (lift + shadow)
- Loading skeletons with shimmer
- Button scale on click

## License

MIT

## Acknowledgments

- Built with [Fastify](https://www.fastify.io/)
- Powered by [OpenAI](https://openai.com/)
- Database by [Neon](https://neon.tech/)
- Redis by [Upstash](https://upstash.com/)
- UI components from [shadcn/ui](https://ui.shadcn.com/)

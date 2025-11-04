# LLM Playground — Next.js + Tailwind + Neon + Auth (Vercel)

All-in-one Next.js app that:
- Streams tokens from OpenAI via SSE (`/api/generate`)
- Logs runs to Neon Postgres via Prisma (`/api/runs`)
- Provides a toy "train" stub (`/api/train`)
- Uses Auth.js (Google) to protect run listing + creation

## ✨ Features

- **SSE Streaming**: Real-time token streaming from OpenAI
- **Authentication**: Google OAuth via NextAuth v5
- **Database**: PostgreSQL (Neon) with Prisma ORM
- **Validation**: Zod schema validation on all API routes
- **Type Safety**: Full TypeScript coverage
- **Error Handling**: Comprehensive error boundaries and user feedback
- **Responsive UI**: Tailwind CSS styling
- **Keyboard Shortcuts**: Cmd/Ctrl + Enter to run
- **Request Cancellation**: AbortController for in-flight requests
- **Security**: Sanitized error messages, auth protection, input validation

## 🚀 Quickstart (Local)

### 1. Clone and Install

```bash
git clone <your-repo>
cd llm-playground-next
npm install
```

### 2. Set Up Environment

```bash
cp .env.local.example .env.local
```

Fill in the required values:
- `OPENAI_API_KEY` - Your OpenAI API key
- `AUTH_SECRET` - Generate with: `openssl rand -base64 32`
- `GOOGLE_CLIENT_ID` - From Google Cloud Console
- `GOOGLE_CLIENT_SECRET` - From Google Cloud Console
- `DATABASE_URL` - Your Neon Postgres connection string (ensure `?sslmode=require`)

**Note:** `AUTH_URL` is optional in development; NextAuth v5 auto-detects on localhost.

### 3. Initialize Database

```bash
# For development (quick setup):
npm run prisma:push

# OR for production (with migrations):
npm run prisma:migrate
```

### 4. Run Development Server

```bash
npm run dev
```

Open http://localhost:3000

## 🔒 Auth Policy

- `/api/runs` **GET** and **POST** require Google sign-in
- To allow anonymous access, remove the `auth()` checks in `/api/runs/route.ts`

## ✅ Validation & Error Handling

- `/api/generate` validates payloads with **Zod** (model, ranges, lengths, max_tokens vs model limits)
- All API routes return **consistent JSON errors**
- Client-side error boundaries with user-friendly messages
- Request cancellation with AbortController
- Sanitized error messages (no internal details leaked)

## 📊 Token Counting & Pricing

- **Token counts are rough estimates** (character-based heuristic: ~4 chars = 1 token)
- For accuracy, integrate server-side tokenization (e.g., `tiktoken`)
- Pricing is configured per-model in `lib/pricing.ts`
- Update rates to match current OpenAI pricing

## 🎨 Adding More Models

Edit `lib/models.ts`:

```ts
export const ALLOWED_MODELS = {
  "gpt-4o-mini": { ctx: 128_000, max_out: 8192 },
  "gpt-4o": { ctx: 200_000, max_out: 16384 }, // Add new models
} as const;
```

Then update `lib/pricing.ts` with corresponding rates.

## 🚀 Deploy to Vercel

### 1. Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Import Project in Vercel

- Go to vercel.com
- Click "Import Project"
- Connect your GitHub repository

### 3. Set Environment Variables

In Vercel Project Settings → Environment Variables, add:
- `OPENAI_API_KEY`
- `AUTH_SECRET`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `DATABASE_URL` (Neon, with `?sslmode=require`)

### 4. Configure Google OAuth

Add these redirect URIs in Google Cloud Console:
- `https://your-domain.vercel.app/api/auth/callback/google`

### 5. Deploy

Vercel automatically runs `prisma generate` during build.

**After first deploy**, run migrations:
```bash
npx prisma migrate deploy
```

Or use `prisma db push` locally once, then redeploy.

## 📁 Project Structure

```
.
├── app/
│   ├── api/
│   │   ├── auth/         # NextAuth routes
│   │   ├── generate/     # SSE streaming endpoint
│   │   ├── runs/         # Run logging (GET/POST)
│   │   └── train/        # Toy training stub
│   ├── layout.tsx        # Root layout with auth
│   ├── page.tsx          # Main playground
│   ├── providers.tsx     # SessionProvider wrapper
│   └── train/            # Training demo page
├── components/           # React components
│   ├── Controls.tsx      # Model parameter controls
│   ├── Output.tsx        # Output display
│   ├── RunsTable.tsx     # Recent runs table
│   └── SignInOut.tsx     # Auth button
├── lib/                  # Utilities and config
│   ├── auth.ts           # NextAuth config
│   ├── env.ts            # Env validation
│   ├── models.ts         # Model whitelist
│   ├── pricing.ts        # Cost estimation
│   ├── prisma.ts         # Prisma client
│   └── types.ts          # TypeScript types
├── prisma/
│   └── schema.prisma     # Database schema
├── public/               # Static assets
└── styles/
    └── globals.css       # Global styles
```

## 🛠️ Scripts

```bash
npm run dev            # Start development server
npm run build          # Build for production
npm run start          # Start production server
npm run prisma:push    # Push schema to database (dev)
npm run prisma:migrate # Create and run migrations
```

## 🐛 Troubleshooting

### Issue: "Missing environment variable"
- Ensure all variables in `.env.local` are set
- Restart dev server after changing env vars

### Issue: "Auth callback error"
- Check Google OAuth redirect URIs include:
  - `http://localhost:3000/api/auth/callback/google` (dev)
  - `https://your-domain.vercel.app/api/auth/callback/google` (prod)

### Issue: "Database connection failed"
- Verify `DATABASE_URL` includes `?sslmode=require` for Neon
- Run `npx prisma db push` to sync schema

### Issue: "Generation failed"
- Verify `OPENAI_API_KEY` is valid
- Check OpenAI API status
- Review browser console for error details

## 🔐 Security Features

- ✅ Auth required for all run operations
- ✅ Input validation with Zod schemas
- ✅ SQL injection protection (Prisma ORM)
- ✅ XSS protection (React escaping)
- ✅ Sanitized error messages
- ✅ CORS configuration
- ✅ Request abortion for cleanup

## 📈 Performance Features

- ✅ SSE streaming for real-time output
- ✅ AbortController for request cancellation
- ✅ Proper cleanup on unmount
- ✅ Efficient state management
- ✅ Graceful database shutdown

## 🎯 Future Enhancements

- [ ] Server-side tiktoken for accurate token counting
- [ ] Rate limiting (Upstash Redis)
- [ ] Multi-turn conversations
- [ ] Cost analytics dashboard
- [ ] Export runs to CSV
- [ ] Dark mode

## 📝 License

MIT

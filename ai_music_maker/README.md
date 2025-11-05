# AI Song Creator

A Next.js application for generating AI-powered song lyrics and audio using OpenAI and Replicate, with Supabase authentication and storage.

## Prerequisites

- Node.js 18+ and npm
- Supabase account and project
- OpenAI API key
- Replicate API token

## Environment Variables

Copy `.env.example` to `.env.local` and fill in your values:

\`\`\`bash
cp .env.example .env.local
\`\`\`

### Supabase

Set the following from your Supabase project settings:

- `NEXT_PUBLIC_SUPABASE_URL` - Your project URL
- `NEXT_PUBLIC_SUPABASE_ANON_KEY` - Your anon/public key
- `SUPABASE_SERVICE_ROLE_KEY` - Your service role key (for server-side operations)

### OpenAI

- `OPENAI_API_KEY` - Your OpenAI API key from https://platform.openai.com/api-keys

### Replicate

- `REPLICATE_API_TOKEN` - Your Replicate API token from https://replicate.com/account/api-tokens
- `REPLICATE_MUSIC_MODEL` - Model version string (default: `meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb`)

### Site URL

- `NEXT_PUBLIC_SITE_URL` - Your site URL (e.g., `http://localhost:3000` for local dev)
- `NEXT_PUBLIC_DEV_SUPABASE_REDIRECT_URL` - Redirect URL for Supabase auth in development

## Setting Environment Variables on Vercel

1. Go to your Vercel project settings
2. Navigate to **Settings** → **Environment Variables**
3. Add each variable from `.env.example`
4. Set the appropriate environment (Production, Preview, Development)
5. Redeploy your application

## Database Setup

Run the SQL migration script in your Supabase SQL editor or via the Supabase CLI:

\`\`\`bash
# The script is located at scripts/001_create_songs_tables.sql
\`\`\`

This creates the `songs` and `song_generations` tables with RLS policies.

## Optional: Generate TypeScript Types

To generate TypeScript types from your Supabase schema:

\`\`\`bash
npx supabase gen types typescript --project-id YOUR_PROJECT_ID > lib/database.types.ts
\`\`\`

Replace `YOUR_PROJECT_ID` with your Supabase project ID.

## Development

Install dependencies and run the development server:

\`\`\`bash
npm install
npm run dev
\`\`\`

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Project Structure

\`\`\`
app/
├── api/generate/          # API routes for lyrics and audio generation
├── auth/                  # Authentication pages
├── songs/                 # Song management pages
components/                # React components
lib/
├── supabase/             # Supabase client utilities
├── env.ts                # Environment variable validation
└── utils.ts              # Utility functions
scripts/                   # Database migration scripts
\`\`\`

## Features

- Email/password authentication with Supabase
- AI-powered lyrics generation using OpenAI
- AI-powered audio generation using Replicate
- Song management dashboard
- Real-time generation status polling
- Glassmorphism UI design

## License

MIT

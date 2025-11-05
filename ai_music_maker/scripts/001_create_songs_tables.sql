-- Enable UUID extension if not already enabled
create extension if not exists "uuid-ossp";

-- Create songs table
create table if not exists public.songs (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text,
  prompt text,
  genre text,
  mood text,
  key text,
  tempo int,
  lyrics text,
  created_at timestamptz default now()
);

-- Create song_generations table
create table if not exists public.song_generations (
  id uuid primary key default uuid_generate_v4(),
  song_id uuid not null references public.songs(id) on delete cascade,
  provider text,
  status text,
  audio_url text,
  raw_response jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Create indexes for better query performance
create index if not exists songs_user_id_idx on public.songs(user_id);
create index if not exists songs_created_at_idx on public.songs(created_at desc);
create index if not exists song_generations_song_id_idx on public.song_generations(song_id);
create index if not exists song_generations_status_idx on public.song_generations(status);

-- Enable Row Level Security
alter table public.songs enable row level security;
alter table public.song_generations enable row level security;

-- RLS Policies for songs table
-- Users can insert their own songs
create policy "Users can insert their own songs"
  on public.songs
  for insert
  to authenticated
  with check (auth.uid() = user_id);

-- Users can select only their own songs
create policy "Users can select their own songs"
  on public.songs
  for select
  to authenticated
  using (auth.uid() = user_id);

-- Users can update only their own songs
create policy "Users can update their own songs"
  on public.songs
  for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- Users can delete only their own songs
create policy "Users can delete their own songs"
  on public.songs
  for delete
  to authenticated
  using (auth.uid() = user_id);

-- RLS Policies for song_generations table
-- Users can insert song_generations for songs they own
create policy "Users can insert song_generations for their songs"
  on public.song_generations
  for insert
  to authenticated
  with check (
    exists (
      select 1 from public.songs
      where songs.id = song_generations.song_id
      and songs.user_id = auth.uid()
    )
  );

-- Users can select song_generations for songs they own
create policy "Users can select song_generations for their songs"
  on public.song_generations
  for select
  to authenticated
  using (
    exists (
      select 1 from public.songs
      where songs.id = song_generations.song_id
      and songs.user_id = auth.uid()
    )
  );

-- Users can update song_generations for songs they own
create policy "Users can update song_generations for their songs"
  on public.song_generations
  for update
  to authenticated
  using (
    exists (
      select 1 from public.songs
      where songs.id = song_generations.song_id
      and songs.user_id = auth.uid()
    )
  )
  with check (
    exists (
      select 1 from public.songs
      where songs.id = song_generations.song_id
      and songs.user_id = auth.uid()
    )
  );

-- Users can delete song_generations for songs they own
create policy "Users can delete song_generations for their songs"
  on public.song_generations
  for delete
  to authenticated
  using (
    exists (
      select 1 from public.songs
      where songs.id = song_generations.song_id
      and songs.user_id = auth.uid()
    )
  );

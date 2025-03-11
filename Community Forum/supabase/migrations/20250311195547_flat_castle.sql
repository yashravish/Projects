/*
  # Fix profiles table RLS policies

  1. Changes
    - Adds RLS policy for inserting new profiles during registration
    - Ensures authenticated users can create their own profile
    - Maintains existing policies for viewing and updating profiles

  2. Security
    - Only allows users to create their own profile
    - Maintains row-level security
*/

-- Drop existing policies to avoid conflicts
DROP POLICY IF EXISTS "Public profiles are viewable by everyone" ON profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON profiles;
DROP POLICY IF EXISTS "Users can insert own profile" ON profiles;

-- Recreate policies with correct permissions
CREATE POLICY "Public profiles are viewable by everyone"
ON profiles FOR SELECT
TO public
USING (true);

CREATE POLICY "Users can update own profile"
ON profiles FOR UPDATE
TO public
USING (auth.uid() = id)
WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
ON profiles FOR INSERT
TO public
WITH CHECK (auth.uid() = id);
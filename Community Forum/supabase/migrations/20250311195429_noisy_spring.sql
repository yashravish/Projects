/*
  # Add vote counting functions

  1. New Functions
    - `get_post_vote_count(post_id)`: Calculates the total vote count for a post
      - Returns the sum of vote values (+1 or -1) for the given post

  2. Changes
    - Creates a function to efficiently calculate post vote counts
    - Function is optimized for performance with parallel execution
*/

CREATE OR REPLACE FUNCTION get_post_vote_count(post_id uuid)
RETURNS integer
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
  SELECT COALESCE(SUM(value), 0)::integer
  FROM votes
  WHERE post_id = $1;
$$;
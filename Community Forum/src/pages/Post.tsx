import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { formatDistanceToNow } from 'date-fns';
import { useAuth } from '../contexts/AuthContext';
import { ArrowUp, ArrowDown } from 'lucide-react';

interface Comment {
  id: string;
  content: string;
  created_at: string;
  author: {
    username: string;
  };
}

interface PostData {
  id: string;
  title: string;
  content: string;
  created_at: string;
  author: {
    username: string;
  };
  votes: {
    sum: number;
  }[];
  vote_count: number;
  user_vote?: number;
}

export default function Post() {
  const { id } = useParams<{ id: string }>();
  const [post, setPost] = useState<PostData | null>(null);
  const [comments, setComments] = useState<Comment[]>([]);
  const [newComment, setNewComment] = useState('');
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();

  useEffect(() => {
    if (id) {
      fetchPost();
      fetchComments();
    }
  }, [id]);

  async function fetchPost() {
    try {
      const { data, error } = await supabase
        .from('posts')
        .select(`
          *,
          author:profiles(username),
          votes:votes(sum:value)
        `)
        .eq('id', id)
        .single();

      if (error) throw error;
      if (data) {
        setPost({
          ...data,
          vote_count: data.votes.reduce((sum, vote) => sum + (vote?.sum || 0), 0)
        });
      }
    } catch (error) {
      console.error('Error fetching post:', error);
    } finally {
      setLoading(false);
    }
  }

  async function fetchComments() {
    try {
      const { data, error } = await supabase
        .from('comments')
        .select(`
          *,
          author:profiles(username)
        `)
        .eq('post_id', id)
        .order('created_at', { ascending: true });

      if (error) throw error;
      if (data) setComments(data);
    } catch (error) {
      console.error('Error fetching comments:', error);
    }
  }

  async function handleSubmitComment(e: React.FormEvent) {
    e.preventDefault();
    if (!user || !newComment.trim()) return;

    try {
      const { error } = await supabase
        .from('comments')
        .insert([
          {
            content: newComment.trim(),
            post_id: id,
            author_id: user.id
          }
        ]);

      if (error) throw error;

      setNewComment('');
      await fetchComments();
    } catch (error) {
      console.error('Error submitting comment:', error);
    }
  }

  async function handleVote(value: number) {
    if (!user || !post) return;

    try {
      const { data: existingVotes } = await supabase
        .from('votes')
        .select('*')
        .eq('post_id', post.id)
        .eq('user_id', user.id);

      const existingVote = existingVotes?.[0];

      if (existingVote) {
        if (existingVote.value === value) {
          await supabase
            .from('votes')
            .delete()
            .eq('post_id', post.id)
            .eq('user_id', user.id);
        } else {
          await supabase
            .from('votes')
            .update({ value })
            .eq('post_id', post.id)
            .eq('user_id', user.id);
        }
      } else {
        await supabase
          .from('votes')
          .insert([{ post_id: post.id, user_id: user.id, value }]);
      }

      await fetchPost();
    } catch (error) {
      console.error('Error voting:', error);
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-400 border-t-transparent shadow-neon"></div>
      </div>
    );
  }

  if (!post) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold text-white">Post not found</h2>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="glass-card p-6 mb-6">
        <div className="flex">
          <div className="flex flex-col items-center mr-4">
            <button
              onClick={() => handleVote(1)}
              className="text-gray-300 hover:text-primary-400 transition-colors"
              aria-label="Upvote"
            >
              <ArrowUp size={20} />
            </button>
            <span className="text-sm font-medium my-1 text-primary-300">{post.vote_count}</span>
            <button
              onClick={() => handleVote(-1)}
              className="text-gray-300 hover:text-primary-400 transition-colors"
              aria-label="Downvote"
            >
              <ArrowDown size={20} />
            </button>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">{post.title}</h1>
            <div className="flex items-center mt-2 text-sm text-gray-400">
              <span>Posted by {post.author.username}</span>
              <span className="mx-2">•</span>
              <span>{formatDistanceToNow(new Date(post.created_at))} ago</span>
            </div>
            <div className="mt-4 text-gray-300 whitespace-pre-wrap leading-relaxed">
              {post.content}
            </div>
          </div>
        </div>
      </div>

      <div className="glass-card p-6">
        <h2 className="text-xl font-semibold mb-4 text-white">Comments</h2>
        
        {user && (
          <form onSubmit={handleSubmitComment} className="mb-6">
            <textarea
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
              placeholder="Write a comment..."
              className="w-full p-3 bg-gray-800/50 border border-gray-700 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-gray-100 placeholder-gray-500"
              rows={3}
            />
            <button
              type="submit"
              disabled={!newComment.trim()}
              className="mt-2 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-500 disabled:opacity-50 transition-colors duration-200"
            >
              Submit
            </button>
          </form>
        )}

        <div className="space-y-4">
          {comments.map((comment) => (
            <div key={comment.id} className="border-b border-gray-700/50 last:border-0 pb-4">
              <div className="flex items-center mb-2">
                <span className="font-medium text-primary-300">{comment.author.username}</span>
                <span className="mx-2 text-gray-600">•</span>
                <span className="text-gray-400 text-sm">
                  {formatDistanceToNow(new Date(comment.created_at))} ago
                </span>
              </div>
              <p className="text-gray-300 leading-relaxed">{comment.content}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
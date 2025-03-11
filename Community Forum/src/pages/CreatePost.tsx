import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { useAuth } from '../contexts/AuthContext';

export default function CreatePost() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [checkingProfile, setCheckingProfile] = useState(true);
  const navigate = useNavigate();
  const { user } = useAuth();

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }

    async function checkProfile() {
      try {
        const { data: profile, error: profileError } = await supabase
          .from('profiles')
          .select('id')
          .eq('id', user.id)
          .single();

        if (profileError || !profile) {
          // Create profile if it doesn't exist
          const { error: createError } = await supabase
            .from('profiles')
            .insert([
              {
                id: user.id,
                username: user.user_metadata.username || `user_${user.id.slice(0, 8)}`,
                updated_at: new Date().toISOString()
              }
            ])
            .single();

          if (createError) {
            throw createError;
          }
        }
      } catch (error) {
        console.error('Error checking profile:', error);
        setError('Unable to verify user profile. Please try signing out and back in.');
      } finally {
        setCheckingProfile(false);
      }
    }

    checkProfile();
  }, [user, navigate]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!user) return;

    setError('');
    setLoading(true);

    try {
      const { data, error: postError } = await supabase
        .from('posts')
        .insert({
          title: title.trim(),
          content: content.trim(),
          author_id: user.id
        })
        .select('id')
        .single();

      if (postError) {
        throw postError;
      }

      if (!data?.id) {
        throw new Error('Failed to create post');
      }

      navigate(`/post/${data.id}`);
    } catch (error: any) {
      console.error('Error creating post:', error);
      setError(error.message || 'Failed to create post. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  if (!user || checkingProfile) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-400 border-t-transparent shadow-neon"></div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="glass-card p-6">
        <h1 className="text-2xl font-bold mb-6 text-white">Create a New Post</h1>
        
        {error && (
          <div className="bg-red-500/20 text-red-300 p-4 rounded-md mb-4 border border-red-500/30">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-gray-300 mb-1">
              Title
            </label>
            <input
              type="text"
              id="title"
              name="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full bg-gray-800 text-white rounded-md border-gray-700 focus:border-primary-500 focus:ring-primary-500 shadow-sm"
              required
            />
          </div>

          <div>
            <label htmlFor="content" className="block text-sm font-medium text-gray-300 mb-1">
              Content
            </label>
            <textarea
              id="content"
              name="content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={6}
              className="w-full bg-gray-800 text-white rounded-md border-gray-700 focus:border-primary-500 focus:ring-primary-500 shadow-sm"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading || !title.trim() || !content.trim()}
            className="w-full py-2 px-4 bg-primary-600 hover:bg-primary-500 text-white font-medium rounded-md shadow-lg hover:shadow-primary-500/20 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 disabled:hover:bg-primary-600 transition-all duration-200"
          >
            {loading ? 'Creating Post...' : 'Create Post'}
          </button>
        </form>
      </div>
    </div>
  );
}
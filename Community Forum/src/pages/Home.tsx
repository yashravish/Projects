import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { formatDistanceToNow } from 'date-fns';
import { ArrowUp, ArrowDown, MessageSquare } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { motion } from 'framer-motion';
import { List, AutoSizer, WindowScroller } from 'react-virtualized';

interface Post {
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
  comments: {
    count: number;
  }[];
  vote_count: number;
  user_vote?: number;
}

export default function Home() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(0);
  const { user } = useAuth();
  const POSTS_PER_PAGE = 10;

  useEffect(() => {
    fetchPosts();
  }, [page]);

  async function fetchPosts() {
    try {
      const { data, error } = await supabase
        .from('posts')
        .select(`
          *,
          author:profiles(username),
          votes:votes(sum:value),
          comments(count)
        `)
        .range(page * POSTS_PER_PAGE, (page + 1) * POSTS_PER_PAGE - 1)
        .order('created_at', { ascending: false });

      if (error) throw error;

      if (data) {
        const newPosts = data.map(post => ({
          ...post,
          vote_count: post.votes.reduce((sum, vote) => sum + (vote?.sum || 0), 0),
          comment_count: post.comments[0].count,
        }));

        setPosts(prevPosts => 
          page === 0 ? newPosts : [...prevPosts, ...newPosts]
        );
        setHasMore(data.length === POSTS_PER_PAGE);
      }
    } catch (error) {
      console.error('Error fetching posts:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleVote(postId: string, value: number) {
    if (!user) return;

    try {
      const { data: existingVotes } = await supabase
        .from('votes')
        .select('*')
        .eq('post_id', postId)
        .eq('user_id', user.id);

      const existingVote = existingVotes?.[0];

      if (existingVote) {
        if (existingVote.value === value) {
          await supabase
            .from('votes')
            .delete()
            .eq('post_id', postId)
            .eq('user_id', user.id);
        } else {
          await supabase
            .from('votes')
            .update({ value })
            .eq('post_id', postId)
            .eq('user_id', user.id);
        }
      } else {
        await supabase
          .from('votes')
          .insert([{ post_id: postId, user_id: user.id, value }]);
      }

      await fetchPosts();
    } catch (error) {
      console.error('Error voting:', error);
    }
  }

  const renderPost = ({ index, key, style }: { index: number; key: string; style: React.CSSProperties }) => {
    const post = posts[index];
    if (!post) return null;

    return (
      <motion.div
        key={key}
        style={style}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: index * 0.1 }}
        className="glass-card card-hover mb-8"
      >
        <div className="p-8">
          <div className="flex items-start gap-6">
            <div className="flex flex-col items-center space-y-2">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleVote(post.id, 1)}
                className="text-gray-300 hover:text-primary-400 transition-colors p-1"
                aria-label="Upvote"
              >
                <ArrowUp size={24} />
              </motion.button>
              <span className="text-lg font-medium neon-text">
                {post.vote_count}
              </span>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleVote(post.id, -1)}
                className="text-gray-300 hover:text-primary-400 transition-colors p-1"
                aria-label="Downvote"
              >
                <ArrowDown size={24} />
              </motion.button>
            </div>
            <div className="flex-1 space-y-4">
              <Link to={`/post/${post.id}`}>
                <h2 className="text-2xl font-bold text-white hover:text-primary-400 transition-colors">
                  {post.title}
                </h2>
              </Link>
              <p className="text-gray-300 leading-relaxed line-clamp-3">{post.content}</p>
              <div className="flex items-center text-sm text-gray-400 space-x-4">
                <span className="font-medium">Posted by {post.author.username}</span>
                <span>•</span>
                <span>{formatDistanceToNow(new Date(post.created_at))} ago</span>
                <div className="flex items-center">
                  <MessageSquare size={16} className="mr-1.5" />
                  <span>{post.comments[0].count} comments</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    );
  };

  if (loading && posts.length === 0) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-400 border-t-transparent shadow-neon"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <WindowScroller>
        {({ height, isScrolling, registerChild, scrollTop }) => (
          <AutoSizer disableHeight>
            {({ width }) => (
              <div ref={registerChild}>
                <List
                  autoHeight
                  height={height}
                  isScrolling={isScrolling}
                  rowCount={posts.length + (hasMore ? 1 : 0)}
                  rowHeight={250}
                  rowRenderer={renderPost}
                  scrollTop={scrollTop}
                  width={width}
                  onRowsRendered={({ stopIndex }) => {
                    if (hasMore && stopIndex === posts.length - 1) {
                      setPage(prev => prev + 1);
                    }
                  }}
                />
              </div>
            )}
          </AutoSizer>
        )}
      </WindowScroller>
    </div>
  );
}
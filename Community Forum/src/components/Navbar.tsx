import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { LogIn, LogOut, PenSquare } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Navbar() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    try {
      await signOut();
      navigate('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <nav className="glass-card sticky top-0 z-50 mb-8">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="text-2xl font-bold neon-text">
            Community Forum
          </Link>

          <div className="flex items-center space-x-4">
            {user ? (
              <>
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Link
                    to="/create-post"
                    className="glass-button flex items-center space-x-2 bg-primary-600 hover:bg-primary-500"
                  >
                    <PenSquare size={18} className="text-white" />
                    <span className="text-white font-medium">New Post</span>
                  </Link>
                </motion.div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleSignOut}
                  className="glass-button flex items-center space-x-2 bg-gray-700 hover:bg-gray-600"
                >
                  <LogOut size={18} className="text-white" />
                  <span className="text-white font-medium">Sign Out</span>
                </motion.button>
              </>
            ) : (
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Link
                  to="/login"
                  className="glass-button flex items-center space-x-2 bg-primary-600 hover:bg-primary-500"
                >
                  <LogIn size={18} className="text-white" />
                  <span className="text-white font-medium">Sign In</span>
                </Link>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
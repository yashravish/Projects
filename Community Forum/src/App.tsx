import React, { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Post from './pages/Post';
import Login from './pages/Login';
import Register from './pages/Register';
import CreatePost from './pages/CreatePost';

function App() {
  useEffect(() => {
    // Create animated background bubbles
    const createBubbles = () => {
      const container = document.body;
      const bubbleCount = 8;

      // Remove existing bubbles
      const existingBubbles = document.querySelectorAll('.bubble');
      existingBubbles.forEach(bubble => bubble.remove());

      // Create new bubbles
      for (let i = 0; i < bubbleCount; i++) {
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        // Random size between 100px and 300px
        const size = Math.random() * 200 + 100;
        bubble.style.width = `${size}px`;
        bubble.style.height = `${size}px`;
        
        // Random position
        bubble.style.left = `${Math.random() * 100}vw`;
        bubble.style.top = `${Math.random() * 100}vh`;
        
        // Random animation delay
        bubble.style.animationDelay = `${Math.random() * 10}s`;
        
        // Random opacity
        bubble.style.opacity = `${Math.random() * 0.3 + 0.1}`;
        
        container.appendChild(bubble);
      }
    };

    createBubbles();
    window.addEventListener('resize', createBubbles);

    return () => {
      window.removeEventListener('resize', createBubbles);
    };
  }, []);

  return (
    <BrowserRouter>
      <AuthProvider>
        <div className="min-h-screen">
          <Navbar />
          <main className="container mx-auto px-4 py-8 relative z-10">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/post/:id" element={<Post />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route path="/create-post" element={<CreatePost />} />
            </Routes>
          </main>
        </div>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
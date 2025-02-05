// src/components/ChatWidget.js
import React from 'react';

const ChatWidget = () => {
  return (
    <div className="chat-widget">
      <button
        className="chat-button"
        onClick={() => alert('Chat functionality coming soon!')}
      >
        Chat with us
      </button>
    </div>
  );
};

export default ChatWidget;

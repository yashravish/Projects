// frontend/src/AgileTaskBoard.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import SockJS from 'sockjs-client';
import { Client } from '@stomp/stompjs';

const AgileTaskBoard = () => {
  const [userStories, setUserStories] = useState([]);
  const [newStory, setNewStory] = useState({ title: '', description: '', status: 'backlog' });

  // Fetch initial user stories from the backend
  useEffect(() => {
    axios.get('/api/userstories')
      .then(response => setUserStories(response.data))
      .catch(err => console.error('Error fetching user stories:', err));
  }, []);

  // Set up the WebSocket connection for live updates
  useEffect(() => {
    const socket = new SockJS('/ws');
    const client = new Client({
      webSocketFactory: () => socket,
      debug: (str) => console.log(str),
      reconnectDelay: 5000,
      onConnect: () => {
        console.log('Connected to WebSocket');
        client.subscribe('/topic/userstories', message => {
          const updatedStory = JSON.parse(message.body);
          setUserStories(prevStories => {
            const index = prevStories.findIndex(story => story.id === updatedStory.id);
            if (index >= 0) {
              // Replace updated story
              const stories = [...prevStories];
              stories[index] = updatedStory;
              return stories;
            } else {
              // Append new story
              return [...prevStories, updatedStory];
            }
          });
        });
      },
    });
    client.activate();

    // Cleanup on unmount
    return () => {
      client.deactivate();
    };
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewStory(prev => ({ ...prev, [name]: value }));
  };

  const createUserStory = () => {
    axios.post('/api/userstories', newStory)
      .then(() => setNewStory({ title: '', description: '', status: 'backlog' }))
      .catch(err => console.error('Error creating user story:', err));
  };

  const renderColumn = (status, title) => (
    <div style={{ flex: 1, margin: '0 10px' }}>
      <h3>{title}</h3>
      {userStories.filter(story => story.status === status).map(story => (
        <div key={story.id} style={{ border: '1px solid #ccc', margin: '5px', padding: '10px' }}>
          <h4>{story.title}</h4>
          <p>{story.description}</p>
        </div>
      ))}
    </div>
  );

  return (
    <div>
      <h1>SprintSync</h1>
      <div>
        <h2>Create User Story</h2>
        <input
          type="text"
          name="title"
          placeholder="Title"
          value={newStory.title}
          onChange={handleInputChange}
        />
        <br />
        <textarea
          name="description"
          placeholder="Description"
          value={newStory.description}
          onChange={handleInputChange}
        />
        <br />
        <select name="status" value={newStory.status} onChange={handleInputChange}>
          <option value="backlog">Backlog</option>
          <option value="in progress">In Progress</option>
          <option value="done">Done</option>
        </select>
        <br />
        <button onClick={createUserStory}>Create</button>
      </div>
      <hr />
      <h2>Sprint Board</h2>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        {renderColumn('backlog', 'Backlog')}
        {renderColumn('in progress', 'In Progress')}
        {renderColumn('done', 'Done')}
      </div>
    </div>
  );
};

export default AgileTaskBoard;

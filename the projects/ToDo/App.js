// App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import { ListGroup, ListGroupItem } from 'reactstrap';

function App() {
  const [tasks, setTasks] = useState([]);

  useEffect(() => {
    axios.get('/fetch_tasks.php')
      .then(response => {
        setTasks(response.data);
      });
  }, []);

  return (
    <div className="App">
      <h1>Todo List</h1>
      <ListGroup>
        {tasks.map(task => (
          <ListGroupItem key={task.id}>
            {task.task} (Due: {task.due_date}, Category: {task.category}, Priority: {task.priority})
          </ListGroupItem>
        ))}
      </ListGroup>
    </div>
  );
}

export default App;
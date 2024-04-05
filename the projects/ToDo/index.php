<?php include 'db.php'; ?>
<!DOCTYPE html>
<html>
<head>
    <title>Todo List Application</title>
    <style>
        /* Add some basic styling */
        body {
            font-family: Arial, sans-serif;
        }
        .task {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <h1>Todo List</h1>
    <div id="taskList"></div>

    <script>
        // Fetch tasks from the database
        fetch('fetch_tasks.php')
            .then(response => response.json())
            .then(tasks => {
                var taskListDiv = document.getElementById('taskList');

                tasks.forEach(function(task) {
                    var taskDiv = document.createElement('div');
                    taskDiv.className = 'task';
                    taskDiv.textContent = task.task + ' (Due: ' + task.due_date + ', Category: ' + task.category + ', Priority: ' + task.priority + ')';
                    taskListDiv.appendChild(taskDiv);
                });
            });
    </script>
</body>
</html>
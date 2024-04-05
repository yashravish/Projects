<?php
include 'db.php';

$sql = "SELECT id, task, due_date, category, priority FROM Tasks";
$result = $conn->query($sql);

$tasks = [];
if ($result->num_rows > 0) {
  // output data of each row
  while($row = $result->fetch_assoc()) {
    $tasks[] = $row;
  }
}

header('Content-Type: application/json');
echo json_encode($tasks);
?>
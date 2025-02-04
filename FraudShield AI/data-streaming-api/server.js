// data-streaming-api/server.js
const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;

// Parse JSON bodies
app.use(bodyParser.json());

app.post('/stream', (req, res) => {
    const transactionData = req.body;
    console.log("Received transaction data for streaming:", transactionData);
    // In a production system, this data might be forwarded to a .NET dashboard via WebSocket or another protocol.
    res.status(200).send({ message: "Data streamed successfully" });
});

app.listen(port, () => {
    console.log(`Data Streaming API running on port ${port}`);
});

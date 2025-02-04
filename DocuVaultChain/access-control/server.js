// access-control/server.js
const express = require('express');
const app = express();
const port = 4000;

app.use(express.json());

// Sample user roles (in a real system, these would be managed via authentication middleware)
const users = {
  "alice": "admin",
  "bob": "client"
};

// Middleware to simulate role-based access
app.use((req, res, next) => {
    const username = req.headers['x-username'];
    if (!username || !users[username]) {
        return res.status(401).json({ message: "Unauthorized: User not found" });
    }
    req.userRole = users[username];
    next();
});

app.get('/access/document/:docId', (req, res) => {
    const { docId } = req.params;
    // Only allow 'admin' role to retrieve any document; clients get limited access.
    if (req.userRole !== 'admin') {
        return res.status(403).json({ message: "Forbidden: Insufficient permissions" });
    }
    res.json({ documentId: docId, content: "Secure Document Content Here" });
});

app.listen(port, () => {
    console.log(`Access Control Service running on port ${port}`);
});

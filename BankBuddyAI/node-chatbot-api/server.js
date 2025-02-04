const express = require('express');
const bodyParser = require('body-parser');
const chatbotRoutes = require('./chatbot_routes');

const app = express();
const port = 3000;

app.use(bodyParser.json());
app.use("/api", chatbotRoutes);

app.listen(port, () => {
  console.log(`BankBuddy Chatbot API is running on port ${port}`);
});

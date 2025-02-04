// chatbot_routes.js
const express = require('express');
const router = express.Router();

// Dummy function to simulate chatbot response
function getChatbotResponse(message) {
  // In production, you might call the Python NLP service via HTTP or use a shared model.
  if (message.toLowerCase().includes("balance")) {
    return "Your current balance is $5,000.";
  } else if (message.toLowerCase().includes("freeze")) {
    return "Your card has been frozen. Please confirm via SMS 2FA.";
  } else {
    return "I'm sorry, I didn't understand that. Could you please rephrase?";
  }
}

router.post("/chat", (req, res) => {
  const { message } = req.body;
  if (!message) {
    return res.status(400).json({ error: "No message provided." });
  }
  const response = getChatbotResponse(message);
  res.json({ response });
});

module.exports = router;

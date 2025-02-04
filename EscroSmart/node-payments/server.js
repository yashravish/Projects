const express = require('express');
const { triggerPayment } = require('./payment_trigger');
const app = express();
const port = 4000;

app.use(express.json());

app.post('/trigger-payment', (req, res) => {
    const { escrowId, amount } = req.body;
    const result = triggerPayment(escrowId, amount);
    res.json(result);
});

app.listen(port, () => {
    console.log(`Payment service running on port ${port}`);
});

package com.fraudshield.api;

public class RiskScore {
    private long transactionId;
    private double score;

    public RiskScore(long transactionId, double score) {
        this.transactionId = transactionId;
        this.score = score;
    }

    // Getters and setters
    public long getTransactionId() {
        return transactionId;
    }
    public void setTransactionId(long transactionId) {
        this.transactionId = transactionId;
    }
    public double getScore() {
        return score;
    }
    public void setScore(double score) {
        this.score = score;
    }
}

package com.fraudshield.api;

public class Transaction {
    private long id;
    private double amount;

    public Transaction() { }

    public Transaction(long id, double amount) {
        this.id = id;
        this.amount = amount;
    }

    // Getters and setters
    public long getId() {
        return id;
    }
    public void setId(long id) {
        this.id = id;
    }
    public double getAmount() {
        return amount;
    }
    public void setAmount(double amount) {
        this.amount = amount;
    }
}

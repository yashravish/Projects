package com.bankbuddy;

public class BankingServices {

    // Simulate card freezing service integration
    public static boolean freezeCard(String cardNumber) {
        // In production, this would involve calling a core banking API.
        System.out.println("Freezing card: " + cardNumber);
        // Simulate successful freezing operation
        return true;
    }

    public static void main(String[] args) {
        String cardNumber = "1234-5678-9012-3456";
        boolean result = freezeCard(cardNumber);
        System.out.println("Card freeze successful: " + result);
    }
}

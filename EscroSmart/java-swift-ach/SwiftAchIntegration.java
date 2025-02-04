package com.escrosmart;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class SwiftAchIntegration {
    // Dummy endpoint URL for SWIFT/ACH network integration.
    private static final String PAYMENT_API_URL = "https://api.paymentnetwork.com/process";

    public static void main(String[] args) {
        try {
            String escrowId = "ESC12345";
            double amount = 50000.00;
            String response = processPayment(escrowId, amount);
            System.out.println("Payment Response: " + response);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String processPayment(String escrowId, double amount) throws Exception {
        URL url = new URL(PAYMENT_API_URL + "?escrowId=" + escrowId + "&amount=" + amount);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setRequestProperty("Accept", "application/json");

        if (conn.getResponseCode() != 200) {
            throw new RuntimeException("Failed: HTTP error code " + conn.getResponseCode());
        }

        BufferedReader br = new BufferedReader(new InputStreamReader((conn.getInputStream())));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
            output.append(line);
        }
        conn.disconnect();
        return output.toString();
    }
}

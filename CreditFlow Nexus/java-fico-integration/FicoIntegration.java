package com.creditflownexus;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class FicoIntegration {

    // Dummy URL for FICO API integration – replace with the real endpoint.
    private static final String FICO_API_URL = "https://api.example.com/fico/score";

    public static void main(String[] args) {
        try {
            String applicantId = "123456";
            double ficoScore = getFicoScore(applicantId);
            System.out.println("Applicant FICO Score: " + ficoScore);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static double getFicoScore(String applicantId) throws Exception {
        URL url = new URL(FICO_API_URL + "?applicantId=" + applicantId);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setRequestProperty("Accept", "application/json");
        
        if (conn.getResponseCode() != 200) {
            throw new RuntimeException("Failed: HTTP error code " + conn.getResponseCode());
        }
        
        BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        StringBuilder response = new StringBuilder();
        String output;
        while ((output = br.readLine()) != null) {
            response.append(output);
        }
        conn.disconnect();
        
        // For demonstration, assume the API returns the score as a plain number string.
        return Double.parseDouble(response.toString());
    }
}

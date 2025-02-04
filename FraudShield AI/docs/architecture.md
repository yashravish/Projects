# FraudShield AI Architecture

## Overview

FraudShield AI is a cloud-native, real-time fraud detection system that integrates multiple technologies:

- **C Module:** Low-latency transaction processing.
- **Python ML Engine:** Adaptive fraud detection model training (TensorFlow).
- **Java/Spring Boot Risk Scoring Service:** Computes risk scores based on transaction data.
- **Node.js Data Streaming API:** Streams data to the dashboard.
- **.NET Dashboard:** Displays real-time fraud alerts to analysts.
- **AWS Lambda/Kinesis:** Provides serverless processing and data ingestion.
- **Jenkins CI/CD:** Automates testing, security scanning, and deployment.
- **PostgreSQL:** (Planned) For robust transaction metadata storage.

## Data Flow

1. **Transaction Processing (C):** Processes and flags transactions.
2. **ML Inference (Python via Lambda):** Applies ML model to incoming data.
3. **Risk Scoring (Java):** Calculates a risk score.
4. **Data Streaming (Node.js):** Streams processed data to the dashboard.
5. **Dashboard (ASP.NET Core):** Visualizes and alerts fraud analysts.

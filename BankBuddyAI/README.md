# BankBuddy AI

**BankBuddy AI** is an AI-powered banking chatbot with transactional capabilities designed to enhance retail banking user experiences and support digital transformation. Key features include:

- **AI-Powered Conversations:** Uses Python and spaCy to train NLP models for understanding customer queries.
- **Conversational APIs:** Exposed via a Node.js/Express service.
- **Core Banking Integration:** Java-based integration (e.g., card freezing).
- **SMS 2FA:** Implemented in .NET using Twilio.
- **Session Security:** Secured using custom C modules.
- **Deployment:** Deployed on AWS Lex and Lambda.
- **Chat History:** Stored in AWS RDS for audit and analytics.

## Project Structure

BankBuddyAI/ ├── python-nlp/ # NLP model training with spaCy ├── node-chatbot-api/ # Conversational API built with Node.js/Express ├── java-core-banking/ # Core banking services integration in Java ├── dotnet-sms-2fa/ # SMS 2FA service using .NET and Twilio ├── c-session-security/ # Session security module in C ├── aws-deployment/ # Deployment scripts/configuration for AWS Lex/Lambda ├── aws-rds/ # SQL script to create the chat history table in AWS RDS └── README.md # Project overview and instructions

ruby
Copy

## How to Get Started

1. **Train the NLP Model:**
   ```bash
   cd python-nlp
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python train_nlp.py
Run the Chatbot API:

bash
Copy
cd node-chatbot-api
npm install
npm start
The API will be available at http://localhost:3000/api/chat.

Build and Run Core Banking Service (Java):

bash
Copy
cd java-core-banking
mvn clean package
java -cp target/core-banking-integration-1.0.0.jar com.bankbuddy.BankingServices
Test SMS 2FA Service (.NET): Open the dotnet-sms-2fa project in Visual Studio or run via command line:

bash
Copy
cd dotnet-sms-2fa
dotnet run
Compile Session Security Module (C):

bash
Copy
cd c-session-security
make
./session_security
Deploy AWS Lambda and Lex Bot:

bash
Copy
cd aws-deployment
chmod +x aws_lambda_deploy.sh
./aws_lambda_deploy.sh
Import lex_configuration.json into AWS Lex to configure your chatbot.

Setup Chat History Table in AWS RDS: Execute the script in your Azure/AWS SQL instance:

sql
Copy
-- Run the commands in aws-rds/create_chat_history.sql
Final Notes
This sample project provides a blueprint for building an AI-powered banking chatbot that integrates NLP, conversational APIs, core banking services, SMS 2FA, and session security. Customize and extend each component to meet production requirements, security standards, and integration needs.

Happy coding and digital banking transformation!

yaml
Copy

---

## Final Remarks

This complete sample project demonstrates how to integrate multiple technologies into a u
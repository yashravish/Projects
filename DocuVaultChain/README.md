# DocuVaultChain

DocuVaultChain is a Secure Document Vault for Custodial Clients featuring Zero-Trust Encryption. It solves custodial document management challenges by integrating:

- **OCR Parsing (Python):** Extracts text from scanned documents.
- **Custom Encryption (C):** Implements zero-trust encryption modules.
- **Cross-Platform Apps (.NET MAUI):** Provides native apps for document access.
- **Access Control Microservices (Node.js):** Manages role-based document access.
- **Document Indexing (Java/Elasticsearch):** Indexes documents for search and retrieval.
- **Metadata Storage (Azure SQL):** Stores document metadata.
- **Cloud Hosting (Azure Blob Storage):** Securely hosts documents.
- **CI/CD & Compliance (CircleCI):** Ensures builds and deployments meet compliance standards.
- **Secure API Endpoints (ASP.NET Core):** Exposes role-based APIs for secure document retrieval.

## Project Structure

DocuVaultChain/ ├── c-encryption/ # Custom C encryption module ├── ocr-parser/ # Python-based OCR parser ├── access-control/ # Node.js microservice for access control ├── document-indexer/ # Java app for indexing documents via Elasticsearch ├── mobile-app/ # .NET MAUI cross-platform application ├── api/ # ASP.NET Core API for secure document retrieval ├── circleci/ # CircleCI configuration for compliance and builds ├── azure-deployment/ # Azure deployment scripts and ARM templates ├── docs/ # Compliance and audit documentation └── README.md # Project overview and instructions

bash
Copy

## How to Get Started

1. **Build the C Encryption Module:**
   ```bash
   cd c-encryption
   make
   ./encryption_module
Run the OCR Parser:

bash
Copy
cd ocr-parser
pip install -r requirements.txt
python ocr_parser.py <path_to_document_image>
Start the Access Control Service:

bash
Copy
cd access-control
npm install
npm start
Build the Document Indexer:

bash
Copy
cd document-indexer
mvn clean package
Build and Run the Mobile App: Open mobile-app/DocuVaultChainApp.csproj in Visual Studio with .NET MAUI support and run the app.

Build and Run the API:

bash
Copy
cd api
dotnet build
dotnet run
Deploy to Azure:

bash
Copy
cd azure-deployment
bash azure_deploy.sh
CircleCI: Configure CircleCI with the provided circleci/config.yml for automated builds and compliance checks.

Compliance & Audit
Refer to the docs/ folder for detailed compliance standards and audit documentation.

License
This project is provided under the MIT License.

yaml
Copy

---

## Final Notes

This complete project demonstrates a multi‑technology solution for a secure document vault
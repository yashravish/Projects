# EscroSmart – Smart Contract-Driven Escrow Automation

EscroSmart automates commercial escrow processes, reducing manual errors and aligning with Citizens' operational efficiency targets. The solution:

- **Implements escrow logic** with Python/Django.
- **Triggers payments** via a Node.js service.
- **Integrates SWIFT/ACH networks** using Java.
- **Provides dashboards** built with .NET Blazor.
- **Validates digital signatures** with a C module.
- **Deploys smart contracts** via Azure Blockchain Workbench.
- **Stores audit trails** in Azure SQL.
- **Ensures compliance** with CircleCI regulatory checks.

## Project Structure

EscroSmart/ ├── django-escrow/ # Django project for escrow logic ├── node-payments/ # Node.js service to trigger payments ├── java-swift-ach/ # Java integration with SWIFT/ACH networks ├── dotnet-dashboard/ # .NET Blazor dashboard for audit trails and monitoring ├── c-signature-validation/ # C module for digital signature validation ├── azure-blockchain/ # Smart contract and deployment scripts for Azure Blockchain ├── azure-sql/ # SQL script to create audit trail table in Azure SQL ├── circleci/ # CircleCI configuration for builds and compliance └── README.md # Project overview and instructions

ruby
Copy

## How to Get Started

1. **Django Escrow Module:**
   ```bash
   cd django-escrow
   pip install -r requirements.txt
   python manage.py migrate
   python manage.py runserver
Use the /api/escrow/create/ endpoint to create and process an escrow record.

Node.js Payment Service:

bash
Copy
cd node-payments
npm install
npm start
The payment trigger API listens on port 4000.

Java SWIFT/ACH Integration:

bash
Copy
cd java-swift-ach
mvn clean package
java -cp target/swift-ach-integration-1.0.0.jar com.escrosmart.SwiftAchIntegration
.NET Blazor Dashboard: Open the dotnet-dashboard/EscroSmartDashboard.sln in Visual Studio and run the Blazor app to view audit trails and dashboard metrics.

C Signature Validation:

bash
Copy
cd c-signature-validation
make
./signature_validation
Deploy Smart Contract:

bash
Copy
cd azure-blockchain
chmod +x deploy_smart_contract.sh
./deploy_smart_contract.sh
Azure SQL – Audit Trail Setup: Run the SQL script azure-sql/create_audit_trail.sql in your Azure SQL Database to create the audit table.

CircleCI: Configure your CircleCI project with the provided circleci/config.yml to automate builds, tests, and regulatory checks.

License
This project is provided under the MIT License.

yaml
Copy

---

## Final Notes

This sample project demonstrates how to integrate multiple technologies into a unified esc
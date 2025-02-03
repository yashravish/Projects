# TestGenX

**TestGenX** is a Microservice Test Automation Framework that auto-generates API tests from an OpenAPI specification. It is designed to be integrated into CI/CD pipelines (via GitHub Actions or Jenkins) and comes with a bonus React dashboard for visualizing test coverage.

## Features

- **Automated Test Generation:**  
  Automatically parse your OpenAPI (Swagger) specification and generate pytest tests that validate your microservice endpoints.
  
- **CI/CD Integration:**  
  Seamlessly integrate test generation and execution into your CI/CD pipelines using GitHub Actions or Jenkins.
  
- **Test Coverage Dashboard (Bonus):**  
  A React dashboard that fetches and displays test coverage metrics.

## Project Structure

TestGenX/
├── generate_tests.py        # Main CLI test generator script
├── openapi_spec.yaml        # Example OpenAPI specification file
├── requirements.txt         # Python dependencies
├── generated_tests/         # (Auto‑generated) test files directory
│   └── test_api.py         # Sample output file (after running the generator)
├── .github/
│   └── workflows/
│         └── test.yml       # GitHub Actions workflow (CI/CD integration)
└── react-dashboard/         # React dashboard project for test coverage
    ├── package.json
    ├── public/
    │   └── index.html
    └── src/
        ├── index.js
        ├── App.js
        └── CoverageDashboard.js


## Prerequisites

### Backend (Test Generation Tool)
- Python 3.7+
- pip

### Frontend (React Dashboard)
- Node.js and npm

## Setup and Usage

### 1. Python Test Generation Tool

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt

Generate Tests:

Run the following command from the project root to generate API tests from your OpenAPI spec:

bash
Copy
python generate_tests.py openapi_spec.yaml --base-url http://localhost:5000
This command will create (or update) the file generated_tests/test_api.py.

Run the Tests:

Execute the tests using pytest:

bash
Copy
pytest generated_tests
2. CI/CD Integration with GitHub Actions
The provided GitHub Actions workflow (in .github/workflows/test.yml) performs the following steps:

Checks out the repository.
Sets up Python.
Installs the dependencies.
Runs the test generation tool.
Executes the generated tests with pytest.
Push your code to trigger the workflow on your main branch or through pull requests.

3. React Dashboard for Test Coverage
Set Up the Dashboard:

Navigate to the react-dashboard directory and install the dependencies:

bash
Copy
cd react-dashboard
npm install
Configure the Coverage API Endpoint:

Ensure your backend or reporting service exposes an API endpoint (e.g., /api/coverage) that returns JSON data with test coverage information. The expected JSON format is:

json
Copy
[
  { "name": "User Service", "coverage": 85 },
  { "name": "Order Service", "coverage": 90 }
]
Run the Dashboard:

Start the React development server:

bash
Copy
npm start
The dashboard will open at http://localhost:3000.
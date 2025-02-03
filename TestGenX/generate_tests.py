#!/usr/bin/env python3
import os
import json
import yaml
import argparse

def load_openapi_spec(file_path):
    """Load an OpenAPI spec from a YAML or JSON file."""
    with open(file_path, "r") as f:
        if file_path.lower().endswith(('.yaml', '.yml')):
            spec = yaml.safe_load(f)
        else:
            spec = json.load(f)
    return spec

def generate_test_code(spec, base_url):
    """Generate test code based on the API paths in the OpenAPI spec."""
    code_lines = []
    code_lines.append("import requests")
    code_lines.append("import pytest")
    code_lines.append("")
    code_lines.append(f"BASE_URL = '{base_url}'")
    code_lines.append("")

    # Iterate over each path and HTTP method in the spec
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            # Create a safe function name from the method and path
            sanitized_path = path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
            if not sanitized_path:
                sanitized_path = "root"
            func_name = f"test_{method.lower()}_{sanitized_path}"
            code_lines.append(f"def {func_name}():")
            # Build the full URL (for simplicity, we assume no path parameters)
            code_lines.append(f"    url = f\"{{BASE_URL}}{path}\"")
            code_lines.append(f"    response = requests.{method.lower()}(url)")
            code_lines.append("    # Basic check: assert that the response status code is 200")
            code_lines.append("    assert response.status_code == 200")
            code_lines.append("")
    return "\n".join(code_lines)

def write_test_file(code, output_dir):
    """Write the generated test code to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "test_api.py")
    with open(file_path, "w") as f:
        f.write(code)
    print(f"Test file generated at: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate API tests from an OpenAPI spec.")
    parser.add_argument("spec_file", help="Path to the OpenAPI spec file (YAML/JSON)")
    parser.add_argument("--base-url", default="http://localhost:5000", help="Base URL of the API")
    parser.add_argument("--output-dir", default="generated_tests", help="Directory for the generated test file")
    args = parser.parse_args()

    spec = load_openapi_spec(args.spec_file)
    test_code = generate_test_code(spec, args.base_url)
    write_test_file(test_code, args.output_dir)

import unittest
import os
import tempfile
from unittest.mock import patch
from cli.static_analyzer import run_static_checks, run_pylint

class TestStaticAnalyzer(unittest.TestCase):
    @patch('subprocess.run')
    def test_run_pylint(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=['pylint'],
            returncode=0,
            stdout="No issues found",
            stderr=""
        )
        result = run_pylint("dummy.py")
        self.assertIn("No issues found", result)

    def test_file_type_handling(self):
        with tempfile.NamedTemporaryFile(suffix=".py") as py_file:
            results = run_static_checks(py_file.name)
            self.assertIn("pylint", results)

        with tempfile.NamedTemporaryFile(suffix=".js") as js_file:
            results = run_static_checks(js_file.name)
            self.assertIn("eslint", results)

        with tempfile.NamedTemporaryFile(suffix=".txt") as txt_file:
            results = run_static_checks(txt_file.name)
            self.assertEqual(results["info"], "No static analysis available for this file type.")

if __name__ == "__main__":
    unittest.main()
import subprocess

def run_static_checks(file_path):
    results = {}
    if file_path.endswith(".py"):
        results["pylint"] = run_pylint(file_path)
        results["bandit"] = run_bandit(file_path)
    elif file_path.endswith(".js"):
        results["eslint"] = run_eslint(file_path)
    else:
        results["info"] = "No static analysis available for this file type."
    return results

def run_pylint(file_path):
    try:
        result = subprocess.run(
            ["pylint", "--score=n", file_path],
            capture_output=True, text=True, check=False
        )
        return result.stdout
    except FileNotFoundError:
        return "pylint not installed"
    except Exception as e:
        return f"Error running pylint: {str(e)}"

def run_bandit(file_path):
    try:
        result = subprocess.run(
            ["bandit", "-q", "-f", "txt", file_path],
            capture_output=True, text=True, check=False
        )
        return result.stdout
    except FileNotFoundError:
        return "bandit not installed"
    except Exception as e:
        return f"Error running bandit: {str(e)}"

def run_eslint(file_path):
    try:
        result = subprocess.run(
            ["eslint", "--format", "stylish", file_path],
            capture_output=True, text=True, check=False
        )
        return result.stdout
    except FileNotFoundError:
        return "eslint not installed"
    except Exception as e:
        return f"Error running eslint: {str(e)}"
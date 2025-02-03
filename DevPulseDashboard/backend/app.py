# backend/app.py
from flask import Flask, jsonify
import random
import datetime

app = Flask(__name__)

def get_pr_turnaround_time():
    """
    Simulate calculation of the average pull request turnaround time in hours.
    Replace with real data from your Git provider.
    """
    return round(random.uniform(4.0, 48.0), 2)

def get_test_pass_rate():
    """
    Simulate a test pass rate (in percentage).
    Replace with real CI/CD data.
    """
    return round(random.uniform(80.0, 100.0), 2)

def get_jira_issue_resolution_time():
    """
    Simulate average Jira issue resolution time in days.
    Replace with real Jira API queries.
    """
    return round(random.uniform(1.0, 10.0), 2)

def aggregate_metrics():
    """
    Aggregate all metrics into a dictionary.
    """
    metrics = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "pr_turnaround_time_hours": get_pr_turnaround_time(),
        "test_pass_rate_percent": get_test_pass_rate(),
        "jira_issue_resolution_days": get_jira_issue_resolution_time(),
        "active_deployments": random.randint(1, 5)  # Example additional metric
    }
    return metrics

@app.route('/api/metrics', methods=['GET'])
def metrics():
    data = aggregate_metrics()
    return jsonify(data)

if __name__ == '__main__':
    # Run on host 0.0.0.0 so the service is accessible externally (for AWS EC2 deployment)
    app.run(host='0.0.0.0', port=5000)

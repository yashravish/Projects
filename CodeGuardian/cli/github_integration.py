import os
import re
import requests

def get_repo_files(repo_path):
    file_list = []
    for root, _, filenames in os.walk(repo_path):
        for filename in filenames:
            if filename.endswith((".py", ".js")):
                rel_path = os.path.relpath(os.path.join(root, filename), repo_path)
                file_list.append(rel_path)
    return file_list

def post_review_comment(repo_url, comment, token, pr_number):
    match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)", repo_url)
    if not match:
        print("Could not parse GitHub repository information from URL.")
        return

    owner = match.group("owner")
    repo = match.group("repo").replace(".git", "")
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "body": comment,
        "event": "COMMENT"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Review comment posted successfully on PR #{pr_number}.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to post review comment: {str(e)}")
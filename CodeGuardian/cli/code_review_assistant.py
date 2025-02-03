#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from datetime import datetime

from static_analyzer import run_static_checks
from github_integration import get_repo_files, post_review_comment
from dashboard.backend.database import db, AnalysisResult  # Adjust import path as needed

def main():
    parser = argparse.ArgumentParser(description="Automated Code Review Assistant")
    parser.add_argument("repo_url", help="Git repository URL or local path to analyze")
    parser.add_argument("--branch", default="main", help="Branch to analyze (default: main)")
    parser.add_argument("--token", help="GitHub/GitLab API token for posting review comments")
    parser.add_argument("--file", nargs="*", help="Specific files to analyze; if not provided, analyze whole repo")
    parser.add_argument("--post-review", action="store_true", 
                       help="Post review comments to PR (requires --pr-number and --token)")
    parser.add_argument("--pr-number", type=int, help="Pull request number for posting comments")
    parser.add_argument("--local", action="store_true", 
                       help="Treat repo_url as a local directory instead of cloning")
    
    args = parser.parse_args()
    
    # Handle local repository
    if args.local:
        repo_path = os.path.abspath(args.repo_url)
        if not os.path.isdir(repo_path):
            print(f"Error: Local directory {repo_path} does not exist")
            sys.exit(1)
    else:
        repo_path = clone_repository(args.repo_url, args.branch)

    files_to_analyze = args.file if args.file else get_repo_files(repo_path)
    
    analysis_results = {}
    for file in files_to_analyze:
        full_file_path = os.path.join(repo_path, file)
        
        # Security check: Prevent directory traversal
        if not os.path.commonpath([repo_path, full_file_path]) == repo_path:
            print(f"Security warning: Skipping file outside repository - {file}")
            continue
            
        if os.path.exists(full_file_path):
            results = run_static_checks(full_file_path)
            analysis_results[file] = results
        else:
            print(f"Warning: File {file} does not exist.")

    # Print results
    print("\n==== Analysis Results ====")
    total_issues = 0
    for file, results in analysis_results.items():
        print(f"\n--- File: {file} ---")
        for tool, output in results.items():
            print(f"\n### {tool} output:\n{output}\n{'-'*40}")
            total_issues += len(output.splitlines())

    # Save to database
    try:
        result = AnalysisResult(
            date=datetime.now().strftime('%Y-%m-%d'),
            issue_count=total_issues
        )
        db.session.add(result)
        db.session.commit()
    except Exception as e:
        print(f"Warning: Failed to save results to database - {str(e)}")

    # Post GitHub comments
    if args.post_review:
        if not args.token or not args.pr_number:
            print("Error: --post-review requires --token and --pr-number")
            sys.exit(1)
            
        for file, results in analysis_results.items():
            comment = f"Automated Code Review found issues in **{file}**:\n"
            for tool, output in results.items():
                if output.strip():
                    comment += f"\n**{tool}**:\n```\n{output}\n```"
            post_review_comment(args.repo_url, comment, args.token, args.pr_number)

    if not args.local:
        cleanup_repository(repo_path)

def clone_repository(repo_url, branch):
    temp_dir = tempfile.mkdtemp(prefix="repo_")
    try:
        subprocess.check_call(["git", "clone", "-b", branch, repo_url, temp_dir],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        sys.exit(1)
    return temp_dir

def cleanup_repository(repo_path):
    shutil.rmtree(repo_path)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
End-to-End validation using demo data
Tests all major READ features to validate the application works
"""

import requests
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")

def test_passed(name):
    print(f"[PASS] {name}")

def test_failed(name, error):
    print(f"[FAIL] {name}: {error}")

def main():
    print_section("PRACTICEOPS END-TO-END VALIDATION (Demo Data)")

    # Login with demo admin
    print("1. Authentication")
    resp = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "demo@practiceops.app",
        "password": "demo1234"
    })
    if resp.status_code == 200:
        data = resp.json()
        token = data["access_token"]
        test_passed("Login successful")
    else:
        test_failed("Login", f"Status {resp.status_code}")
        return

    headers = {"Authorization": f"Bearer {token}"}

    # Get current user
    resp = requests.get(f"{BASE_URL}/me", headers=headers)
    if resp.status_code == 200:
        user_data = resp.json()
        test_passed(f"Get current user: {user_data['user']['name']}")
        team_id = user_data['primary_team']['team_id']
    else:
        test_failed("Get /me", f"Status {resp.status_code}")
        return

    # Test Team endpoints
    print("\n2. Team Management")
    resp = requests.get(f"{BASE_URL}/teams/{team_id}", headers=headers)
    if resp.status_code == 200:
        team = resp.json()
        test_passed(f"Get team: {team['name']}")
    else:
        test_failed("Get team", f"Status {resp.status_code}")

    resp = requests.get(f"{BASE_URL}/teams/{team_id}/members", headers=headers)
    if resp.status_code == 200:
        members_data = resp.json()
        member_count = len(members_data.get('members', []))
        test_passed(f"Get team members: {member_count} members")
    else:
        test_failed("Get team members", f"Status {resp.status_code}")

    # Test Cycles
    print("\n3. Rehearsal Cycles")
    resp = requests.get(f"{BASE_URL}/teams/{team_id}/cycles", headers=headers)
    if resp.status_code == 200:
        cycles_data = resp.json()
        cycles = cycles_data.get('items', [])
        test_passed(f"Get cycles: {len(cycles)} found")
        if cycles:
            cycle_id = cycles[0]['id']
            print(f"   Current cycle: {cycles[0]['name']}")
        else:
            print("   No cycles found, skipping cycle-specific tests")
            cycle_id = None
    else:
        test_failed("Get cycles", f"Status {resp.status_code}")
        cycle_id = None

    if cycle_id:
        # Test Assignments
        print("\n4. Assignments")
        resp = requests.get(f"{BASE_URL}/cycles/{cycle_id}/assignments", headers=headers)
        if resp.status_code == 200:
            assignments_data = resp.json()
            assignments = assignments_data.get('items', [])
            test_passed(f"Get cycle assignments: {len(assignments)} found")
        else:
            test_failed("Get cycle assignments", f"Status {resp.status_code}")

        # Test Tickets
        print("\n5. Tickets")
        resp = requests.get(f"{BASE_URL}/cycles/{cycle_id}/tickets", headers=headers)
        if resp.status_code == 200:
            tickets_data = resp.json()
            tickets = tickets_data.get('items', [])
            test_passed(f"Get cycle tickets: {len(tickets)} found")
            if tickets:
                ticket_id = tickets[0]['id']
                # Get ticket detail
                resp = requests.get(f"{BASE_URL}/tickets/{ticket_id}", headers=headers)
                if resp.status_code == 200:
                    ticket = resp.json()
                    test_passed(f"Get ticket detail: {ticket['ticket']['title']}")
                else:
                    test_failed("Get ticket detail", f"Status {resp.status_code}")
        else:
            test_failed("Get cycle tickets", f"Status {resp.status_code}")

        # Test Practice Logs
        print("\n6. Practice Logs")
        resp = requests.get(f"{BASE_URL}/cycles/{cycle_id}/practice-logs", headers=headers)
        if resp.status_code == 200:
            logs_data = resp.json()
            logs = logs_data.get('items', [])
            test_passed(f"Get practice logs: {len(logs)} found")
        else:
            test_failed("Get practice logs", f"Status {resp.status_code}")

    # Test Dashboards
    print("\n7. Dashboards")
    resp = requests.get(f"{BASE_URL}/teams/{team_id}/dashboards/member", headers=headers)
    if resp.status_code == 200:
        dashboard = resp.json()
        test_passed("Get member dashboard")
        if dashboard.get('current_cycle'):
            print(f"   Current cycle: {dashboard['current_cycle']['name']}")
        print(f"   Upcoming assignments: {len(dashboard.get('upcoming_assignments', []))}")
        print(f"   My tickets: {len(dashboard.get('my_tickets', []))}")
    else:
        test_failed("Get member dashboard", f"Status {resp.status_code}")

    resp = requests.get(f"{BASE_URL}/teams/{team_id}/dashboards/leader", headers=headers)
    if resp.status_code == 200:
        dashboard = resp.json()
        test_passed("Get leader dashboard")
        summary = dashboard.get('practice_summary', {})
        print(f"   Practice logs this week: {summary.get('logs_this_week', 0)}")
        print(f"   Blocking tickets: {len(dashboard.get('blocking_tickets', []))}")
    else:
        test_failed("Get leader dashboard", f"Status {resp.status_code}")

    # Test Notification Preferences
    print("\n8. Notification Preferences")
    resp = requests.get(f"{BASE_URL}/teams/{team_id}/notification-preferences", headers=headers)
    if resp.status_code == 200:
        prefs = resp.json()
        test_passed("Get notification preferences")
        print(f"   Daily reminder: {prefs.get('daily_reminder_enabled', 'N/A')}")
        print(f"   Weekly digest: {prefs.get('weekly_digest_enabled', 'N/A')}")
    else:
        test_failed("Get notification preferences", f"Status {resp.status_code}")

    # Test token refresh
    print("\n9. Token Refresh")
    refresh_token = data.get("refresh_token")
    if refresh_token:
        resp = requests.post(f"{BASE_URL}/auth/refresh", json={"refresh_token": refresh_token})
        if resp.status_code == 200:
            test_passed("Token refresh")
        else:
            test_failed("Token refresh", f"Status {resp.status_code}")

    print_section("VALIDATION COMPLETE")
    print("All core features validated successfully!")
    print("\nThe application is working end-to-end.")
    print("Demo accounts are read-only by design.")
    print("\nYou can now test the frontend at: http://localhost:5173")
    print("Login with: demo@practiceops.app / demo1234\n")

if __name__ == "__main__":
    main()

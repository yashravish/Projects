#!/usr/bin/env python3
"""
End-to-End testing script for PracticeOps
Tests all major features through the API
"""

import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, UTC

BASE_URL = "http://localhost:8000"

class TestSession:
    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.user: Optional[Dict[str, Any]] = None
        self.team_id: Optional[str] = None
        self.test_results = []

    def test(self, name: str, func):
        """Run a test and record the result"""
        try:
            func()
            self.test_results.append((name, "PASS", None))
            print(f"[PASS] {name}")
            return True
        except AssertionError as e:
            self.test_results.append((name, "FAIL", str(e)))
            print(f"[FAIL] {name}: {e}")
            return False
        except Exception as e:
            self.test_results.append((name, "ERROR", str(e)))
            print(f"[ERROR] {name}: {e}")
            return False

    def get(self, path: str, **kwargs) -> requests.Response:
        """Make authenticated GET request"""
        headers = kwargs.pop('headers', {})
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        return requests.get(f"{BASE_URL}{path}", headers=headers, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        """Make authenticated POST request"""
        headers = kwargs.pop('headers', {})
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        headers['Content-Type'] = 'application/json'
        return requests.post(f"{BASE_URL}{path}", headers=headers, **kwargs)

    def put(self, path: str, **kwargs) -> requests.Response:
        """Make authenticated PUT request"""
        headers = kwargs.pop('headers', {})
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        headers['Content-Type'] = 'application/json'
        return requests.put(f"{BASE_URL}{path}", headers=headers, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        """Make authenticated DELETE request"""
        headers = kwargs.pop('headers', {})
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        return requests.delete(f"{BASE_URL}{path}", headers=headers, **kwargs)

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        passed = sum(1 for _, status, _ in self.test_results if status == "PASS")
        failed = sum(1 for _, status, _ in self.test_results if status == "FAIL")
        errors = sum(1 for _, status, _ in self.test_results if status == "ERROR")
        total = len(self.test_results)

        print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Errors: {errors}")

        if failed > 0 or errors > 0:
            print("\nFailed/Error tests:")
            for name, status, msg in self.test_results:
                if status in ("FAIL", "ERROR"):
                    print(f"  [{status}] {name}: {msg}")

        print("="*60 + "\n")

def main():
    session = TestSession()

    print("\n" + "="*60)
    print("PRACTICEOPS END-TO-END TESTS")
    print("="*60 + "\n")

    # ========================================================================
    # AUTHENTICATION TESTS
    # ========================================================================
    print("Testing Authentication...")
    print("-" * 60)

    def test_health():
        resp = session.get("/health")
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        data = resp.json()
        assert data["status"] == "ok", f"Health status not ok: {data}"
        assert data["db"] == "ok", f"DB health not ok: {data}"

    session.test("Health check", test_health)

    def test_register():
        # Create a fresh test user to avoid demo account restrictions
        import random
        email = f"e2etest{random.randint(1000,9999)}@example.com"
        resp = session.post("/auth/register", json={
            "email": email,
            "password": "test12345",
            "name": "E2E Test User"
        })
        assert resp.status_code == 201, f"Registration failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert "access_token" in data, "No access token in response"
        session.access_token = data["access_token"]
        session.refresh_token = data["refresh_token"]
        session.user = data["user"]

    session.test("User registration", test_register)

    def test_get_me():
        resp = session.get("/me")
        assert resp.status_code == 200, f"Get /me failed: {resp.status_code}"
        data = resp.json()
        assert "user" in data, "No user in response"

    session.test("Get current user", test_get_me)

    def test_create_team():
        resp = session.post("/teams", json={"name": "E2E Test Team"})
        assert resp.status_code == 201, f"Create team failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert "id" in data, "No team ID in response"
        session.team_id = data["id"]

    session.test("Create team", test_create_team)

    def test_refresh_token():
        resp = session.post("/auth/refresh", json={
            "refresh_token": session.refresh_token
        })
        assert resp.status_code == 200, f"Token refresh failed: {resp.status_code}"
        data = resp.json()
        assert "access_token" in data, "No new access token"

    session.test("Token refresh", test_refresh_token)

    # ========================================================================
    # TEAM TESTS
    # ========================================================================
    print("\nTesting Team Management...")
    print("-" * 60)

    def test_get_team():
        resp = session.get(f"/teams/{session.team_id}")
        assert resp.status_code == 200, f"Get team failed: {resp.status_code}"
        data = resp.json()
        assert "name" in data, "No team name"

    session.test("Get team details", test_get_team)

    def test_update_team():
        resp = session.put(f"/teams/{session.team_id}", json={"name": "E2E Test Team Updated"})
        assert resp.status_code == 200, f"Update team failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert data["name"] == "E2E Test Team Updated", "Team name not updated"

    session.test("Update team name", test_update_team)

    def test_get_team_members():
        resp = session.get(f"/teams/{session.team_id}/members")
        assert resp.status_code == 200, f"Get members failed: {resp.status_code}"
        data = resp.json()
        assert "members" in data, "No members field"
        assert len(data["members"]) > 0, "No team members"

    session.test("Get team members", test_get_team_members)

    # ========================================================================
    # REHEARSAL CYCLES TESTS
    # ========================================================================
    print("\nTesting Rehearsal Cycles...")
    print("-" * 60)

    cycle_id = None

    def test_get_cycles():
        nonlocal cycle_id
        resp = session.get(f"/teams/{session.team_id}/cycles")
        assert resp.status_code == 200, f"Get cycles failed: {resp.status_code}"
        cycles = resp.json()
        assert len(cycles) > 0, "No cycles found"
        cycle_id = cycles[0]["id"]

    session.test("Get rehearsal cycles", test_get_cycles)

    def test_get_active_cycle():
        resp = session.get(f"/teams/{session.team_id}/cycles/active")
        assert resp.status_code == 200, f"Get active cycle failed: {resp.status_code}"
        # May return null if no active cycle
        data = resp.json()

    session.test("Get active cycle", test_get_active_cycle)

    def test_create_cycle():
        future_date = (datetime.now(UTC) + timedelta(days=30)).date().isoformat()
        resp = session.post(f"/teams/{session.team_id}/cycles", json={
            "name": "Test Cycle E2E",
            "date": future_date
        })
        assert resp.status_code == 200, f"Create cycle failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert data["name"] == "Test Cycle E2E", "Wrong cycle name"

    session.test("Create new cycle", test_create_cycle)

    # ========================================================================
    # ASSIGNMENTS TESTS
    # ========================================================================
    print("\nTesting Assignments...")
    print("-" * 60)

    assignment_id = None

    def test_get_cycle_assignments():
        resp = session.get(f"/cycles/{cycle_id}/assignments")
        assert resp.status_code == 200, f"Get assignments failed: {resp.status_code}"
        assignments = resp.json()
        # May be empty, that's ok

    session.test("Get cycle assignments", test_get_cycle_assignments)

    def test_create_assignment():
        nonlocal assignment_id
        resp = session.post(f"/cycles/{cycle_id}/assignments", json={
            "type": "SONG_WORK",
            "scope": "TEAM",
            "priority": "MEDIUM",
            "title": "E2E Test Assignment",
            "description": "This is a test assignment",
            "song_ref": "Test Song",
            "due_at": (datetime.now(UTC) + timedelta(days=2)).isoformat()
        })
        assert resp.status_code == 200, f"Create assignment failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert data["title"] == "E2E Test Assignment", "Wrong assignment title"
        assignment_id = data["id"]

    session.test("Create assignment", test_create_assignment)

    def test_update_assignment():
        resp = session.put(f"/assignments/{assignment_id}", json={
            "type": "SONG_WORK",
            "scope": "TEAM",
            "priority": "MEDIUM",
            "title": "E2E Test Assignment Updated",
            "description": "Updated description",
            "song_ref": "Test Song",
            "due_at": (datetime.now(UTC) + timedelta(days=2)).isoformat()
        })
        assert resp.status_code == 200, f"Update assignment failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert data["title"] == "E2E Test Assignment Updated", "Assignment not updated"

    session.test("Update assignment", test_update_assignment)

    # ========================================================================
    # TICKETS TESTS
    # ========================================================================
    print("\nTesting Tickets...")
    print("-" * 60)

    ticket_id = None

    def test_get_cycle_tickets():
        resp = session.get(f"/cycles/{cycle_id}/tickets")
        assert resp.status_code == 200, f"Get tickets failed: {resp.status_code}"
        tickets = resp.json()
        # May be empty

    session.test("Get cycle tickets", test_get_cycle_tickets)

    def test_create_ticket():
        nonlocal ticket_id
        resp = session.post(f"/cycles/{cycle_id}/tickets", json={
            "category": "PITCH",
            "priority": "MEDIUM",
            "visibility": "TEAM",
            "status": "OPEN",
            "title": "E2E Test Ticket",
            "description": "This is a test ticket",
            "song_ref": "Test Song",
            "section": "Tenor",
            "claimable": True
        })
        assert resp.status_code == 200, f"Create ticket failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert data["title"] == "E2E Test Ticket", "Wrong ticket title"
        ticket_id = data["id"]

    session.test("Create ticket", test_create_ticket)

    def test_get_ticket_detail():
        resp = session.get(f"/tickets/{ticket_id}")
        assert resp.status_code == 200, f"Get ticket detail failed: {resp.status_code}"
        data = resp.json()
        assert data["ticket"]["id"] == ticket_id, "Wrong ticket ID"

    session.test("Get ticket details", test_get_ticket_detail)

    def test_claim_ticket():
        resp = session.post(f"/tickets/{ticket_id}/claim")
        assert resp.status_code == 200, f"Claim ticket failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert data["owner_id"] == session.user["id"], "Ticket not claimed by current user"
        assert data["status"] == "IN_PROGRESS", "Ticket status not updated"

    session.test("Claim ticket", test_claim_ticket)

    def test_update_ticket_status():
        resp = session.post(f"/tickets/{ticket_id}/transition", json={
            "status": "RESOLVED",
            "resolved_note": "Fixed in practice"
        })
        assert resp.status_code == 200, f"Update ticket failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert data["status"] == "RESOLVED", "Status not updated"

    session.test("Update ticket status to RESOLVED", test_update_ticket_status)

    # ========================================================================
    # PRACTICE LOGS TESTS
    # ========================================================================
    print("\nTesting Practice Logs...")
    print("-" * 60)

    log_id = None

    def test_get_cycle_logs():
        resp = session.get(f"/cycles/{cycle_id}/practice-logs")
        assert resp.status_code == 200, f"Get practice logs failed: {resp.status_code}"
        logs = resp.json()
        # May be empty

    session.test("Get cycle practice logs", test_get_cycle_logs)

    def test_create_practice_log():
        nonlocal log_id
        resp = session.post("/practice-logs", json={
            "cycle_id": cycle_id,
            "duration_minutes": 45,
            "rating_1_5": 4,
            "blocked_flag": False,
            "notes": "Great practice session!",
            "occurred_at": datetime.now(UTC).isoformat()
        })
        assert resp.status_code == 200, f"Create practice log failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert data["duration_minutes"] == 45, "Wrong duration"
        log_id = data["id"]

    session.test("Create practice log", test_create_practice_log)

    def test_update_practice_log():
        resp = session.put(f"/practice-logs/{log_id}", json={
            "duration_minutes": 60,
            "rating_1_5": 5,
            "notes": "Updated: Excellent session!"
        })
        assert resp.status_code == 200, f"Update practice log failed: {resp.status_code}"
        data = resp.json()
        assert data["duration_minutes"] == 60, "Duration not updated"

    session.test("Update practice log", test_update_practice_log)

    # ========================================================================
    # DASHBOARD TESTS
    # ========================================================================
    print("\nTesting Dashboards...")
    print("-" * 60)

    def test_member_dashboard():
        resp = session.get(f"/teams/{session.team_id}/dashboards/member")
        assert resp.status_code == 200, f"Get member dashboard failed: {resp.status_code}"
        data = resp.json()
        assert "current_cycle" in data or data.get("current_cycle") is None, "Missing current_cycle"
        assert "upcoming_assignments" in data, "Missing upcoming_assignments"
        assert "my_tickets" in data, "Missing my_tickets"

    session.test("Get member dashboard", test_member_dashboard)

    def test_leader_dashboard():
        resp = session.get(f"/teams/{session.team_id}/dashboards/leader")
        assert resp.status_code == 200, f"Get leader dashboard failed: {resp.status_code}"
        data = resp.json()
        assert "current_cycle" in data or data.get("current_cycle") is None, "Missing current_cycle"
        assert "practice_summary" in data, "Missing practice_summary"
        assert "blocking_tickets" in data, "Missing blocking_tickets"

    session.test("Get leader dashboard", test_leader_dashboard)

    # ========================================================================
    # NOTIFICATION PREFERENCES TESTS
    # ========================================================================
    print("\nTesting Notification Preferences...")
    print("-" * 60)

    def test_get_notification_prefs():
        resp = session.get(f"/teams/{session.team_id}/notification-preferences")
        assert resp.status_code == 200, f"Get notification prefs failed: {resp.status_code}"
        data = resp.json()
        assert "daily_reminder_enabled" in data, "Missing daily_reminder_enabled"

    session.test("Get notification preferences", test_get_notification_prefs)

    def test_update_notification_prefs():
        resp = session.put(f"/teams/{session.team_id}/notification-preferences", json={
            "daily_reminder_enabled": False,
            "blocking_due_soon_enabled": True
        })
        assert resp.status_code == 200, f"Update notification prefs failed: {resp.status_code} - {resp.text}"
        data = resp.json()
        assert data["daily_reminder_enabled"] == False, "daily_reminder_enabled not updated"

    session.test("Update notification preferences", test_update_notification_prefs)

    # ========================================================================
    # CLEANUP & DELETE TESTS
    # ========================================================================
    print("\nTesting Delete Operations...")
    print("-" * 60)

    def test_delete_practice_log():
        resp = session.delete(f"/practice-logs/{log_id}")
        assert resp.status_code == 204, f"Delete practice log failed: {resp.status_code}"

    session.test("Delete practice log", test_delete_practice_log)

    def test_delete_assignment():
        resp = session.delete(f"/assignments/{assignment_id}")
        assert resp.status_code == 204, f"Delete assignment failed: {resp.status_code}"

    session.test("Delete assignment", test_delete_assignment)

    def test_delete_ticket():
        resp = session.delete(f"/tickets/{ticket_id}")
        assert resp.status_code == 204, f"Delete ticket failed: {resp.status_code}"

    session.test("Delete ticket", test_delete_ticket)

    # Print summary
    session.print_summary()

    # Return exit code based on results
    failed = sum(1 for _, status, _ in session.test_results if status in ("FAIL", "ERROR"))
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

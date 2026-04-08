"""Playwright UI tests for the history page."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.ui]

FRONTEND_URL = "http://localhost:8080"


class TestHistoryUI:
    """Playwright tests for the capture history page (history.html)."""

    def test_history_page_loads(self, page: Page):
        """Verify history page loads with correct heading."""
        page.goto(f"{FRONTEND_URL}/history.html")
        expect(page).to_have_title("History — Clinical Imaging QA Lab")
        expect(page.locator("h1")).to_have_text("Capture History")

    def test_history_table_present(self, page: Page):
        """Verify history table has expected column headers."""
        page.goto(f"{FRONTEND_URL}/history.html")
        page.wait_for_selector("#history-body", timeout=10000)
        headers = page.locator("thead th")
        texts = headers.all_text_contents()
        assert "Patient" in texts
        assert "Status" in texts
        assert "Type" in texts

    def test_history_table_has_rows(self, page: Page):
        """Verify history table renders rows (assumes captures exist)."""
        page.goto(f"{FRONTEND_URL}/history.html")
        page.wait_for_selector("#history-body tr", timeout=10000)
        rows = page.locator("#history-body tr")
        count = rows.count()
        assert count >= 1

    def test_history_status_badges(self, page: Page):
        """Verify captures table uses status badges."""
        page.goto(f"{FRONTEND_URL}/history.html")
        page.wait_for_selector("#history-body .badge", timeout=10000)
        badges = page.locator("#history-body .badge")
        assert badges.count() >= 1

    def test_history_refresh_button(self, page: Page):
        """Verify refresh button reloads the table."""
        page.goto(f"{FRONTEND_URL}/history.html")
        page.wait_for_selector("#history-body tr td", timeout=10000)
        page.locator("button:has-text('Refresh')").click()
        page.wait_for_selector("#history-body tr td", timeout=10000)
        rows = page.locator("#history-body tr")
        assert rows.count() >= 1

    def test_history_responsive_table(self, page: Page):
        """Verify table renders properly at mobile viewport."""
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto(f"{FRONTEND_URL}/history.html")
        page.wait_for_selector("#history-body", timeout=10000)
        table = page.locator(".table-container")
        expect(table).to_be_visible()

"""Playwright UI tests for the dashboard page."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.ui]

FRONTEND_URL = "http://localhost:8080"


class TestDashboardUI:
    """Playwright tests for the dashboard (index.html)."""

    def test_dashboard_loads(self, page: Page):
        """Verify dashboard page loads with correct title and heading."""
        page.goto(FRONTEND_URL)
        expect(page).to_have_title("Dashboard — Clinical Imaging QA Lab")
        expect(page.locator("h1")).to_have_text("Dashboard")

    def test_dashboard_has_navigation(self, page: Page):
        """Verify all navigation links are present."""
        page.goto(FRONTEND_URL)
        nav = page.locator("nav")
        expect(nav.get_by_role("menuitem", name="Dashboard")).to_be_visible()
        expect(nav.get_by_role("menuitem", name="Capture")).to_be_visible()
        expect(nav.get_by_role("menuitem", name="History")).to_be_visible()
        expect(nav.get_by_role("menuitem", name="Defects")).to_be_visible()

    def test_dashboard_shows_device_status(self, page: Page):
        """Verify device status indicator appears on dashboard."""
        page.goto(FRONTEND_URL)
        page.wait_for_selector(".device-status", timeout=10000)
        status_area = page.locator("#device-status-area .device-status")
        expect(status_area).to_be_visible()

    def test_dashboard_shows_stat_cards(self, page: Page):
        """Verify statistic cards render on the dashboard."""
        page.goto(FRONTEND_URL)
        page.wait_for_selector(".stat-card", timeout=10000)
        cards = page.locator(".stat-card")
        expect(cards).to_have_count(4)

    def test_dashboard_navigation_to_capture(self, page: Page):
        """Verify clicking Capture nav link navigates correctly."""
        page.goto(FRONTEND_URL)
        page.get_by_role("menuitem", name="Capture").click()
        expect(page).to_have_url(f"{FRONTEND_URL}/capture.html")

    def test_dashboard_navigation_to_history(self, page: Page):
        """Verify clicking History nav link navigates correctly."""
        page.goto(FRONTEND_URL)
        page.get_by_role("menuitem", name="History").click()
        expect(page).to_have_url(f"{FRONTEND_URL}/history.html")

    def test_dashboard_has_recent_tables(self, page: Page):
        """Verify recent captures and defects tables are present."""
        page.goto(FRONTEND_URL)
        page.wait_for_selector("#recent-captures-body", timeout=10000)
        expect(page.locator("#recent-captures-body")).to_be_visible()
        expect(page.locator("#recent-defects-body")).to_be_visible()

    def test_dashboard_skip_link(self, page: Page):
        """Verify skip-to-main-content link exists and is accessible."""
        page.goto(FRONTEND_URL)
        skip = page.locator(".skip-link")
        expect(skip).to_have_attribute("href", "#main-content")

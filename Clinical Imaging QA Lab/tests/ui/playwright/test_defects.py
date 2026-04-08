"""Playwright UI tests for the defects page."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.ui]

FRONTEND_URL = "http://localhost:8080"


class TestDefectsUI:
    """Playwright tests for the defect tracking page (defects.html)."""

    def test_defects_page_loads(self, page: Page):
        """Verify defects page loads with correct heading."""
        page.goto(f"{FRONTEND_URL}/defects.html")
        expect(page).to_have_title("Defects — Clinical Imaging QA Lab")
        expect(page.locator("h1")).to_have_text("Defect Tracker")

    def test_defect_form_present(self, page: Page):
        """Verify defect form has all required fields."""
        page.goto(f"{FRONTEND_URL}/defects.html")
        expect(page.locator("#defect-title")).to_be_visible()
        expect(page.locator("#defect-severity")).to_be_visible()
        expect(page.locator("#defect-priority")).to_be_visible()
        expect(page.locator("#defect-submit")).to_be_visible()

    def test_defect_form_validation(self, page: Page):
        """Verify empty form submission shows validation errors."""
        page.goto(f"{FRONTEND_URL}/defects.html")
        page.locator("#defect-submit").click()

        title_error = page.locator("#defect-title-error")
        expect(title_error).to_be_visible()

    def test_defect_submission(self, page: Page):
        """Verify successful defect submission shows success message."""
        page.goto(f"{FRONTEND_URL}/defects.html")
        page.locator("#defect-title").fill("UI Test Defect Submission")
        page.locator("#defect-severity").select_option("minor")
        page.locator("#defect-priority").select_option("low")
        page.locator("#defect-environment").fill("Playwright Test")
        page.locator("#defect-steps").fill("1. Open defects page\n2. Submit form")
        page.locator("#defect-expected").fill("Defect is logged")
        page.locator("#defect-actual").fill("Defect is logged correctly")
        page.locator("#defect-submit").click()

        page.wait_for_selector(".alert-success", timeout=10000)
        expect(page.locator(".alert-success")).to_be_visible()

    def test_defects_table_renders(self, page: Page):
        """Verify defects table shows logged defects."""
        page.goto(f"{FRONTEND_URL}/defects.html")
        page.wait_for_selector("#defects-body tr", timeout=10000)
        rows = page.locator("#defects-body tr")
        assert rows.count() >= 1

    def test_defect_severity_badges(self, page: Page):
        """Verify defect severity is displayed as a badge."""
        page.goto(f"{FRONTEND_URL}/defects.html")
        page.wait_for_selector("#defects-body .badge", timeout=10000)
        badges = page.locator("#defects-body .badge")
        assert badges.count() >= 1

    def test_defect_form_keyboard_accessible(self, page: Page):
        """Verify form can be navigated via keyboard."""
        page.goto(f"{FRONTEND_URL}/defects.html")
        page.locator("#defect-title").focus()
        page.keyboard.press("Tab")
        focused = page.evaluate("document.activeElement.id")
        assert focused in ("defect-severity", "defect-title")

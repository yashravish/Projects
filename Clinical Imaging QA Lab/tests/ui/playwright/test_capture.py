"""Playwright UI tests for the capture page."""
import pytest
import httpx
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.ui]

FRONTEND_URL = "http://localhost:8080"
DEVICE_URL = "http://localhost:8001"


class TestCaptureUI:
    """Playwright tests for the capture workflow (capture.html)."""

    def test_capture_page_loads(self, page: Page):
        """Verify capture page loads with correct heading."""
        page.goto(f"{FRONTEND_URL}/capture.html")
        expect(page).to_have_title("Capture — Clinical Imaging QA Lab")
        expect(page.locator("h1")).to_have_text("Image Capture")

    def test_capture_form_has_required_fields(self, page: Page):
        """Verify all capture form fields are present."""
        page.goto(f"{FRONTEND_URL}/capture.html")
        expect(page.locator("#patient-id")).to_be_visible()
        expect(page.locator("#session-id")).to_be_visible()
        expect(page.locator("#image-type")).to_be_visible()
        expect(page.locator("#capture-submit")).to_be_visible()

    def test_capture_form_labels_associated(self, page: Page):
        """Verify form labels are properly associated with inputs."""
        page.goto(f"{FRONTEND_URL}/capture.html")
        label = page.locator('label[for="patient-id"]')
        expect(label).to_be_visible()
        expect(label).to_contain_text("Patient ID")

    def test_capture_form_validation_empty(self, page: Page):
        """Verify submitting empty form shows validation errors."""
        page.goto(f"{FRONTEND_URL}/capture.html")
        page.locator("#capture-submit").click()

        patient_error = page.locator("#patient-id-error")
        expect(patient_error).to_be_visible()

    def test_capture_successful_submission(self, page: Page):
        """Verify successful capture shows success alert."""
        httpx.post(f"{DEVICE_URL}/device/reset", timeout=5.0)

        page.goto(f"{FRONTEND_URL}/capture.html")
        page.locator("#patient-id").fill("PAT-PW-001")
        page.locator("#session-id").fill("SESS-PW-001")
        page.locator("#image-type").select_option("x-ray")
        page.locator("#capture-submit").click()

        page.wait_for_selector("#capture-result .alert", timeout=15000)
        result = page.locator("#capture-result .alert")
        expect(result).to_be_visible()

    def test_capture_with_device_offline(self, page: Page):
        """Verify capture failure shows error alert when device is offline."""
        httpx.post(f"{DEVICE_URL}/device/disconnect", timeout=5.0)

        page.goto(f"{FRONTEND_URL}/capture.html")
        page.locator("#patient-id").fill("PAT-PW-OFFLINE")
        page.locator("#session-id").fill("SESS-PW-OFFLINE")
        page.locator("#image-type").select_option("mri")
        page.locator("#capture-submit").click()

        page.wait_for_selector("#capture-result .alert-danger", timeout=15000)
        result = page.locator("#capture-result .alert-danger")
        expect(result).to_be_visible()

        httpx.post(f"{DEVICE_URL}/device/reconnect", timeout=5.0)

    def test_capture_retry_button_appears_on_failure(self, page: Page):
        """Verify retry button appears for failed captures."""
        httpx.post(f"{DEVICE_URL}/device/disconnect", timeout=5.0)

        page.goto(f"{FRONTEND_URL}/capture.html")
        page.locator("#patient-id").fill("PAT-PW-RETRY")
        page.locator("#session-id").fill("SESS-PW-RETRY")
        page.locator("#image-type").select_option("ct-scan")
        page.locator("#capture-submit").click()

        page.wait_for_selector("#capture-result .alert-danger", timeout=15000)
        retry_btn = page.locator("#capture-result button")
        expect(retry_btn).to_be_visible()

        httpx.post(f"{DEVICE_URL}/device/reconnect", timeout=5.0)

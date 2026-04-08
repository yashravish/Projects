"""Accessibility tests using Playwright and axe-core."""
import pytest
from playwright.sync_api import Page

pytestmark = [pytest.mark.accessibility]

FRONTEND_URL = "http://localhost:8080"

AXE_SCRIPT_URL = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"


def run_axe(page: Page) -> dict:
    """Inject axe-core and run accessibility checks, returning the results."""
    page.evaluate(
        """() => {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = '%s';
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }"""
        % AXE_SCRIPT_URL
    )
    results = page.evaluate("() => axe.run()")
    return results


class TestAccessibility:
    """Accessibility checks for all pages using axe-core."""

    def test_dashboard_accessibility(self, page: Page):
        """Check dashboard for accessibility violations."""
        page.goto(FRONTEND_URL)
        page.wait_for_selector("h1", timeout=10000)
        results = run_axe(page)
        violations = results.get("violations", [])
        critical = [v for v in violations if v["impact"] in ("critical", "serious")]
        assert len(critical) == 0, (
            f"Found {len(critical)} critical/serious accessibility violations:\n"
            + "\n".join(f"- {v['id']}: {v['description']}" for v in critical)
        )

    def test_capture_accessibility(self, page: Page):
        """Check capture page for accessibility violations."""
        page.goto(f"{FRONTEND_URL}/capture.html")
        page.wait_for_selector("h1", timeout=10000)
        results = run_axe(page)
        violations = results.get("violations", [])
        critical = [v for v in violations if v["impact"] in ("critical", "serious")]
        assert len(critical) == 0, (
            f"Found {len(critical)} critical/serious accessibility violations:\n"
            + "\n".join(f"- {v['id']}: {v['description']}" for v in critical)
        )

    def test_history_accessibility(self, page: Page):
        """Check history page for accessibility violations."""
        page.goto(f"{FRONTEND_URL}/history.html")
        page.wait_for_selector("h1", timeout=10000)
        results = run_axe(page)
        violations = results.get("violations", [])
        critical = [v for v in violations if v["impact"] in ("critical", "serious")]
        assert len(critical) == 0, (
            f"Found {len(critical)} critical/serious accessibility violations:\n"
            + "\n".join(f"- {v['id']}: {v['description']}" for v in critical)
        )

    def test_defects_accessibility(self, page: Page):
        """Check defects page for accessibility violations."""
        page.goto(f"{FRONTEND_URL}/defects.html")
        page.wait_for_selector("h1", timeout=10000)
        results = run_axe(page)
        violations = results.get("violations", [])
        critical = [v for v in violations if v["impact"] in ("critical", "serious")]
        assert len(critical) == 0, (
            f"Found {len(critical)} critical/serious accessibility violations:\n"
            + "\n".join(f"- {v['id']}: {v['description']}" for v in critical)
        )

    def test_form_labels_associated(self, page: Page):
        """Verify all form inputs have associated labels."""
        page.goto(f"{FRONTEND_URL}/capture.html")
        inputs = page.locator("input[required], select[required]")
        count = inputs.count()
        for i in range(count):
            input_el = inputs.nth(i)
            input_id = input_el.get_attribute("id")
            label = page.locator(f'label[for="{input_id}"]')
            assert label.count() == 1, f"Input #{input_id} missing associated label"

    def test_focus_visible_styles(self, page: Page):
        """Verify interactive elements have visible focus styling."""
        page.goto(f"{FRONTEND_URL}/capture.html")
        page.locator("#patient-id").focus()
        box_shadow = page.locator("#patient-id").evaluate(
            "el => getComputedStyle(el).boxShadow"
        )
        assert box_shadow != "none", "Focused input should have visible focus style"

    def test_skip_link_present(self, page: Page):
        """Verify skip navigation link exists on all pages."""
        for path in ["", "/capture.html", "/history.html", "/defects.html"]:
            page.goto(f"{FRONTEND_URL}{path}")
            skip = page.locator(".skip-link")
            assert skip.count() == 1, f"Skip link missing on {path or '/'}"

    def test_landmarks_present(self, page: Page):
        """Verify semantic landmarks are present."""
        page.goto(FRONTEND_URL)
        assert page.locator("header[role='banner']").count() == 1
        assert page.locator("main").count() == 1
        assert page.locator("footer[role='contentinfo']").count() == 1
        assert page.locator("nav[aria-label]").count() >= 1

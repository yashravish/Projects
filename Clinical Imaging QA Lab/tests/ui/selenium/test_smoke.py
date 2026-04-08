"""Selenium smoke tests for cross-browser validation."""
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

pytestmark = [pytest.mark.smoke, pytest.mark.ui]

FRONTEND_URL = "http://localhost:8080"


def get_chrome_driver():
    """Create a headless Chrome WebDriver instance."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def get_firefox_driver():
    """Create a headless Firefox WebDriver instance."""
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    return webdriver.Firefox(options=options)


class TestSmokeChromeSelenium:
    """Selenium smoke tests using Chrome."""

    @pytest.fixture(autouse=True)
    def setup_driver(self):
        self.driver = get_chrome_driver()
        yield
        self.driver.quit()

    def test_dashboard_loads_chrome(self):
        """Verify dashboard page loads in Chrome."""
        self.driver.get(FRONTEND_URL)
        assert "Dashboard" in self.driver.title
        h1 = self.driver.find_element(By.TAG_NAME, "h1")
        assert h1.text == "Dashboard"

    def test_navigation_links_chrome(self):
        """Verify navigation links are present in Chrome."""
        self.driver.get(FRONTEND_URL)
        nav_links = self.driver.find_elements(By.CSS_SELECTOR, ".nav-links a")
        link_texts = [link.text for link in nav_links]
        assert "Dashboard" in link_texts
        assert "Capture" in link_texts
        assert "History" in link_texts
        assert "Defects" in link_texts

    def test_capture_page_chrome(self):
        """Verify capture page loads with form in Chrome."""
        self.driver.get(f"{FRONTEND_URL}/capture.html")
        assert "Capture" in self.driver.title
        form = self.driver.find_element(By.ID, "capture-form")
        assert form.is_displayed()

    def test_capture_form_submission_chrome(self):
        """Verify capture form can be submitted in Chrome."""
        self.driver.get(f"{FRONTEND_URL}/capture.html")
        wait = WebDriverWait(self.driver, 10)

        patient_input = wait.until(EC.presence_of_element_located((By.ID, "patient-id")))
        patient_input.send_keys("PAT-SEL-001")

        session_input = self.driver.find_element(By.ID, "session-id")
        session_input.send_keys("SESS-SEL-001")

        from selenium.webdriver.support.ui import Select
        image_select = Select(self.driver.find_element(By.ID, "image-type"))
        image_select.select_by_value("x-ray")

        submit_btn = self.driver.find_element(By.ID, "capture-submit")
        submit_btn.click()

        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#capture-result .alert"))
        )
        result = self.driver.find_element(By.CSS_SELECTOR, "#capture-result .alert")
        assert result.is_displayed()

    def test_history_page_chrome(self):
        """Verify history page loads in Chrome."""
        self.driver.get(f"{FRONTEND_URL}/history.html")
        assert "History" in self.driver.title
        wait = WebDriverWait(self.driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, "history-body")))

    def test_defects_page_chrome(self):
        """Verify defects page loads in Chrome."""
        self.driver.get(f"{FRONTEND_URL}/defects.html")
        assert "Defects" in self.driver.title
        form = self.driver.find_element(By.ID, "defect-form")
        assert form.is_displayed()


class TestSmokeFirefoxSelenium:
    """Selenium smoke tests using Firefox."""

    @pytest.fixture(autouse=True)
    def setup_driver(self):
        try:
            self.driver = get_firefox_driver()
            yield
            self.driver.quit()
        except Exception:
            pytest.skip("Firefox WebDriver not available")
            yield

    def test_dashboard_loads_firefox(self):
        """Verify dashboard page loads in Firefox."""
        self.driver.get(FRONTEND_URL)
        assert "Dashboard" in self.driver.title

    def test_capture_page_firefox(self):
        """Verify capture page loads in Firefox."""
        self.driver.get(f"{FRONTEND_URL}/capture.html")
        assert "Capture" in self.driver.title

    def test_history_page_firefox(self):
        """Verify history page loads in Firefox."""
        self.driver.get(f"{FRONTEND_URL}/history.html")
        assert "History" in self.driver.title

    def test_defects_page_firefox(self):
        """Verify defects page loads in Firefox."""
        self.driver.get(f"{FRONTEND_URL}/defects.html")
        assert "Defects" in self.driver.title

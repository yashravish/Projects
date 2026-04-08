"""Playwright configuration for pytest-playwright."""
import pytest


@pytest.fixture(scope="session")
def browser_type_launch_args():
    """Launch browser in headless mode."""
    return {"headless": True}


@pytest.fixture(scope="session")
def browser_context_args():
    """Configure browser context settings."""
    return {
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }

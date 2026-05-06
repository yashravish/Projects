import os
import time
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")

def main() -> None:
    print(f"RiskFlow Python worker ready; API_BASE_URL={API_BASE_URL}")
    while True:
        try:
            response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
            print(f"health={response.status_code}")
        except requests.RequestException as exc:
            print(f"waiting for API: {exc}")
        time.sleep(60)

if __name__ == "__main__":
    main()

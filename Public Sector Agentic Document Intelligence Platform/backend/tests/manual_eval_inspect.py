"""Quick inspection of the most-recent eval run from inside the api container."""
from __future__ import annotations

import httpx


def main() -> None:
    with httpx.Client(timeout=120.0) as c:
        r = c.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"email": "seed-admin@example.gov", "password": "ChangeMe!2026"},
        )
        r.raise_for_status()
        h = {"Authorization": "Bearer " + r.json()["access_token"]}

        runs = c.get("http://localhost:8000/api/v1/evaluations", headers=h).json()
        if not runs["items"]:
            print("(no eval runs on file)")
            return
        rid = runs["items"][0]["run_id"]
        detail = c.get(
            f"http://localhost:8000/api/v1/evaluations/{rid}", headers=h
        ).json()
        for i, it in enumerate(detail["items"][:3]):
            inq = it["inquiry"]
            print(f"--- {i+1}. {it['gold']['id']} ---")
            print("Q:", it["gold"]["question"])
            print("A:", inq["answer_text"][:480])
            print(
                "RECALL:",
                f"{it['metrics']['retrieval_recall']:.2f}",
                "| FAITH:",
                f"{it['metrics']['faithfulness']:.2f}",
                "| GROUND:",
                f"{it['metrics']['grounding_score']:.2f}",
            )
            if inq["citations"]:
                print(
                    "CITE:",
                    inq["citations"][0]["document_filename"],
                    "p",
                    inq["citations"][0]["page_start"],
                )
                print(" ", inq["citations"][0]["snippet"][:200])
            print()


if __name__ == "__main__":
    main()

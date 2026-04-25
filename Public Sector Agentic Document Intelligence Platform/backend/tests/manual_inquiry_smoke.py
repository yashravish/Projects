"""Manual smoke driver — run from inside the api container.

Drives login → /query/inquiry → /query/runs → /query/runs/{id} and prints
the deliverables. Used to bypass the Windows-host port-mapping wedge during
development; not part of pytest collection.
"""
from __future__ import annotations

import sys

import httpx

BASE = "http://localhost:8000/api/v1"


def main() -> int:
    with httpx.Client(timeout=60.0) as c:
        r = c.post(
            f"{BASE}/auth/login",
            json={"email": "seed-admin@example.gov", "password": "ChangeMe!2026"},
        )
        r.raise_for_status()
        token = r.json()["access_token"]
        h = {"Authorization": f"Bearer {token}"}

        question = "What is the deadline for the Resilient Communities grant?"
        r = c.post(
            f"{BASE}/query/inquiry",
            headers=h,
            json={"question": question},
        )
        if r.status_code != 200:
            print("STATUS", r.status_code)
            print("BODY", r.text)
            return 1
        body = r.json()

        print("STATUS    :", body["status"])
        print("MODEL     :", body["model"])
        print("LATENCY MS:", body["latency_ms"])
        print("RETRIEVED :", len(body["retrieved"]))
        print("CITATIONS :", len(body["citations"]))
        print("TRACE     :", " -> ".join(t["node"] for t in body["trace"]))
        print(
            "CRITIQUE  :",
            f"grounding={body['critique']['grounding_score']:.2f}",
            f"hallucination={body['critique']['hallucination_risk']:.2f}",
            f"passed={body['critique']['passed']}",
        )
        print("MLFLOW RUN:", body.get("mlflow_run_id"))
        print()
        print("ANSWER:")
        print(body["answer_text"])
        print()
        print("CITATIONS:")
        for cit in body["citations"]:
            print(
                f"  [{cit['index']}] {cit['document_filename']}"
                f" pages {cit['page_start']}-{cit['page_end']}"
            )
            print(f"      {cit['snippet'][:160]}…")

        runs = c.get(f"{BASE}/query/runs", headers=h).json()
        print(f"\nHISTORY total={runs['total']}")
        for item in runs["items"][:5]:
            print(
                f"  {item['created_at']}  passes={item['grounding_score']}",
                f"  {item['question'][:60]}",
            )

        detail = c.get(f"{BASE}/query/runs/{body['run_id']}", headers=h).json()
        same = detail["answer_text"] == body["answer_text"]
        print(f"\nDETAIL replay matches: {same}")
        return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())

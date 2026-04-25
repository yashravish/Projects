"""Manual smoke driver — run from inside the api container.

Drives login → POST /evaluations/run → GET /evaluations → GET /evaluations/{id}
and prints the aggregate readings + a few per-item rows. Used to bypass the
Windows-host port-mapping wedge during development; not part of pytest
collection.
"""
from __future__ import annotations

import sys

import httpx

BASE = "http://localhost:8000/api/v1"


def main() -> int:
    with httpx.Client(timeout=180.0) as c:
        r = c.post(
            f"{BASE}/auth/login",
            json={"email": "seed-admin@example.gov", "password": "ChangeMe!2026"},
        )
        r.raise_for_status()
        token = r.json()["access_token"]
        h = {"Authorization": f"Bearer {token}"}

        ds = c.get(f"{BASE}/evaluations/dataset", headers=h)
        ds.raise_for_status()
        dataset = ds.json()
        print(f"DATASET: {dataset['name']} v{dataset['version']} ({dataset['n_items']} items)")

        r = c.post(f"{BASE}/evaluations/run", headers=h, json={})
        if r.status_code != 200:
            print("STATUS", r.status_code)
            print("BODY", r.text)
            return 1
        body = r.json()

        agg = body["aggregate"]
        print()
        print(f"RUN     : {body['run_id']}")
        print(f"STATUS  : {body['status']}")
        print(f"MODEL   : {body['model']}")
        print(f"WALL    : {body['wall_time_ms']} ms")
        print(f"MLFLOW  : {body.get('mlflow_run_id')}")
        print()
        print("AGGREGATE")
        for k in (
            "n_items",
            "pass_rate",
            "retrieval_recall",
            "retrieval_precision",
            "citation_precision",
            "citation_recall",
            "faithfulness",
            "forbidden_phrase_rate",
            "grounding_score",
            "hallucination_risk",
            "latency_ms_p50",
            "latency_ms_p95",
            "n_failures",
        ):
            print(f"  {k:<22}: {agg[k]}")

        print()
        print("PER-ITEM")
        for it in body["items"]:
            m = it["metrics"]
            mark = "PASS" if m["item_passed"] else "FAIL"
            print(
                f"  [{mark}] {it['gold']['id']:<22}"
                f" recall={m['retrieval_recall']:.2f}"
                f" faith={m['faithfulness']:.2f}"
                f" ground={m['grounding_score']:.2f}"
                f" hr={m['hallucination_risk']:.2f}"
                f" lat={m['latency_ms']}ms"
            )

        # History
        runs = c.get(f"{BASE}/evaluations", headers=h).json()
        print()
        print(f"HISTORY total={runs['total']}")
        for item in runs["items"][:5]:
            print(
                f"  {item['created_at']}  pass={item['pass_rate']:.2f}"
                f"  recall={item['retrieval_recall']:.2f}"
                f"  {item['dataset_name']}"
            )

        # Detail replay
        detail = c.get(f"{BASE}/evaluations/{body['run_id']}", headers=h).json()
        same = (
            detail["aggregate"]["n_items"] == body["aggregate"]["n_items"]
            and len(detail["items"]) == len(body["items"])
        )
        print(f"\nDETAIL replay matches: {same}")
        return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())

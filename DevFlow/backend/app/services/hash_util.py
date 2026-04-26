import hashlib


def stable_user_bucket(user_id: str, mod: int = 10_000) -> int:
    h = hashlib.sha256(f"devflow:{user_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def stable_variant_choice(user_id: str, a_percent: int) -> str:
    """Return 'A' or 'B' with deterministic split. a_percent in [0,100]."""
    a_percent = max(0, min(100, a_percent))
    b = stable_user_bucket(user_id, 100)
    if b < a_percent:
        return "A"
    return "B"

from app.services.hash_util import stable_user_bucket


def is_flag_granted(rollout_percentage: int, user_id: str, flag_id: int, enabled: bool) -> bool:
    if not enabled or rollout_percentage <= 0:
        return False
    bucket = stable_user_bucket(f"{user_id}:flag:{flag_id}", 100)
    return bucket < rollout_percentage

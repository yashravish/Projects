"""Static checks on the gold dataset.

The dataset is shipped as a Python module so this test does double duty:
it asserts the data is internally consistent AND that the version hash is
deterministic (a property tests rely on for reproducibility).
"""
from __future__ import annotations

import pytest

from app.eval.dataset import GOLD_DATASET, GoldQuestionDataset, get_dataset


def test_dataset_is_non_empty() -> None:
    assert len(GOLD_DATASET) >= 5, "gold dataset should cover the corpus"


def test_dataset_version_is_stable() -> None:
    # Same content → same version. Reconstruct from the same items and confirm.
    duplicate = GoldQuestionDataset(
        name=GOLD_DATASET.name,
        description=GOLD_DATASET.description,
        items=GOLD_DATASET.items,
    )
    assert duplicate.version == GOLD_DATASET.version


def test_dataset_version_changes_when_content_changes() -> None:
    mutated_items = (*GOLD_DATASET.items[1:],)  # drop the first item
    mutated = GoldQuestionDataset(
        name=GOLD_DATASET.name,
        description=GOLD_DATASET.description,
        items=mutated_items,
    )
    assert mutated.version != GOLD_DATASET.version


def test_every_item_has_well_formed_required_fields() -> None:
    seen_ids: set[str] = set()
    for item in GOLD_DATASET:
        assert item.id, "id must not be empty"
        assert item.id not in seen_ids, f"duplicate item id {item.id!r}"
        seen_ids.add(item.id)
        assert item.question.strip(), f"{item.id} has empty question"
        assert item.expected_doc_filenames, (
            f"{item.id} has no expected_doc_filenames"
        )
        for fname in item.expected_doc_filenames:
            assert fname.endswith(".pdf"), f"{item.id} expected_doc {fname!r}"
        assert item.must_contain_any, (
            f"{item.id} has no must_contain_any (faithfulness untestable)"
        )
        # Each OR-group must contain at least one phrase.
        for group in item.must_contain_any:
            assert group, f"{item.id} has empty OR-group"
            for phrase in group:
                assert phrase.strip(), f"{item.id} has empty phrase"


def test_get_dataset_default_returns_gold() -> None:
    assert get_dataset(None) is GOLD_DATASET
    assert get_dataset(GOLD_DATASET.name) is GOLD_DATASET


def test_get_dataset_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        get_dataset("not-a-real-dataset")


def test_dataset_targets_only_seeded_filenames() -> None:
    """Sanity: the dataset must reference filenames produced by
    `app.seed.generate_sample_pdfs.build_sample_pdfs`. Otherwise the
    retrieval recall metric is unreachable on a fresh seed."""
    from app.seed.generate_sample_pdfs import build_sample_pdfs

    seeded = {p.filename for p in build_sample_pdfs()}
    referenced = {
        fname for item in GOLD_DATASET for fname in item.expected_doc_filenames
    }
    missing = referenced - seeded
    assert not missing, (
        f"gold dataset references unseeded filenames: {sorted(missing)}; "
        f"seeded: {sorted(seeded)}"
    )

import json
import os


SPLIT_PATH = "experiments/splits/split_indices.json"


def load_split():
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_split_file_exists():
    assert os.path.exists(SPLIT_PATH)


def test_split_contains_required_keys():
    split = load_split()

    required_keys = [
        "random_seed",
        "shuffle",
        "stratify",
        "train_indices",
        "dev_indices",
        "test_indices",
    ]

    for key in required_keys:
        assert key in split


def test_split_no_overlap():
    split = load_split()

    train_indices = set(split["train_indices"])
    dev_indices = set(split["dev_indices"])
    test_indices = set(split["test_indices"])

    assert train_indices.isdisjoint(dev_indices)
    assert train_indices.isdisjoint(test_indices)
    assert dev_indices.isdisjoint(test_indices)


def test_split_has_no_duplicates():
    split = load_split()

    for key in ["train_indices", "dev_indices", "test_indices"]:
        indices = split[key]
        assert len(indices) == len(set(indices))


def test_split_is_not_empty():
    split = load_split()

    assert len(split["train_indices"]) > 0
    assert len(split["dev_indices"]) > 0
    assert len(split["test_indices"]) > 0


def test_split_is_fixed_seed_42():
    split = load_split()

    assert split["random_seed"] == 42
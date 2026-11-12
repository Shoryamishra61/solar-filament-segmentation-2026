from datetime import UTC, datetime

import numpy as np

from .artifacts import object_sha256


def make_folds(manifest: dict, n_splits: int = 5, seed: int = 2026) -> dict:
    if not 3 <= n_splits <= 10:
        raise ValueError("Use between three and ten folds")
    observations = manifest["train"]
    parents = {row["file_name"]: row["file_name"] for row in observations}

    def find(name: str) -> str:
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    month_seen, pixels_seen = {}, {}
    for row in observations:
        name = row["file_name"]
        for registry, key in [
            (month_seen, row["captured"][:7]),
            (pixels_seen, row["pixel_sha256"]),
        ]:
            if key in registry:
                parents[find(name)] = find(registry[key])
            else:
                registry[key] = name
    groups: dict[str, list[str]] = {}
    for name in sorted(parents):
        groups.setdefault(find(name), []).append(name)
    if len(groups) < n_splits:
        raise ValueError("Not enough independent month/duplicate groups for the requested folds")
    rng = np.random.default_rng(seed)
    shuffled = list(groups.values())
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    sizes = [0] * n_splits
    assignments = {}
    for names in shuffled:
        fold = min(range(n_splits), key=lambda index: (sizes[index], index))
        sizes[fold] += len(names)
        assignments.update(dict.fromkeys(names, fold))
    return {
        "schema_version": 1,
        "manifest_sha256": object_sha256(manifest),
        "policy": "calendar-month-and-exact-decoded-duplicates",
        "seed": seed,
        "n_splits": n_splits,
        "holdout_fold": 0,
        "assignments": assignments,
        "fold_sizes": sizes,
    }


def validate_folds(manifest: dict, folds: dict) -> None:
    if (
        type(folds.get("holdout_fold")) is not int
        or folds["holdout_fold"] != 0
        or type(folds.get("n_splits")) is not int
    ):
        raise ValueError("The reserved holdout contract is invalid")
    if not 3 <= folds["n_splits"] <= 10:
        raise ValueError("The number of folds is invalid")
    if folds["manifest_sha256"] != object_sha256(manifest):
        raise ValueError("Fold manifest does not match the frozen dataset")
    assignments = folds["assignments"]
    if set(assignments) != {row["file_name"] for row in manifest["train"]}:
        raise ValueError("Fold assignments must cover every training photograph exactly")
    if any(type(f) is not int or not 0 <= f < folds["n_splits"] for f in assignments.values()):
        raise ValueError("Invalid fold assignment")
    for key in ["pixel_sha256", "month"]:
        seen = {}
        for row in manifest["train"]:
            group = row["captured"][:7] if key == "month" else row[key]
            fold = assignments[row["file_name"]]
            if group in seen and seen[group] != fold:
                raise ValueError(f"Leakage: {key} group crosses folds")
            seen[group] = fold
    if any(not any(f == fold for f in assignments.values()) for fold in range(folds["n_splits"])):
        raise ValueError("A validation fold is empty")


def partition(
    manifest: dict, folds: dict, validation_fold: int, embargo_hours: float = 72.0
) -> tuple[list[str], list[str]]:
    validate_folds(manifest, folds)
    if type(validation_fold) is not int or not 0 <= validation_fold < folds["n_splits"]:
        raise ValueError("Invalid validation fold")
    if not np.isfinite(embargo_hours) or embargo_hours < 0:
        raise ValueError("Embargo hours must be finite and nonnegative")
    assignments = folds["assignments"]
    excluded = {validation_fold, folds["holdout_fold"]}
    protected_times = np.array(
        [
            datetime.fromisoformat(row["captured"]).replace(tzinfo=UTC).timestamp()
            for row in manifest["train"]
            if assignments[row["file_name"]] in excluded
        ]
    )
    training, validation = [], []
    for row in manifest["train"]:
        name = row["file_name"]
        if assignments[name] == validation_fold:
            validation.append(name)
        elif assignments[name] not in excluded:
            captured = datetime.fromisoformat(row["captured"]).replace(tzinfo=UTC).timestamp()
            if np.abs(protected_times - captured).min() >= embargo_hours * 3600:
                training.append(name)
    if not training or not validation:
        raise ValueError("The fold/embargo policy leaves an empty training or validation set")
    return sorted(training), sorted(validation)

"""Reproduce the candidate: 85% general ensemble, 15% platform specialists."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from bot_detection.model import feature_sets, ensemble_predict, specialist_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("candidate.csv"))
    parser.add_argument("--feature-cache", type=Path, help="Optional trusted local feature cache created by experiments/improve.py")
    args = parser.parse_args()
    train = pd.read_csv(args.data / "train.csv")
    test = pd.read_csv(args.data / "test.csv")
    meta = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    assert meta.cookie_id.is_unique and train.target.isin([0, 1]).all()
    if args.feature_cache:
        # A local speed option only: no predictions or labels are loaded from the cache.
        cached = pickle.loads(args.feature_cache.read_bytes())
        fingerprint = {name: hashlib.sha256((args.data/name).read_bytes()).hexdigest()
                       for name in ["train.csv", "test.csv", "events.csv.gz"]}
        assert cached["fingerprint"] == fingerprint, "Feature cache belongs to different inputs"
        frames = cached["frames"]
    else:
        events = pd.read_csv(args.data / "events.csv.gz", parse_dates=["event_ts"])
        frames = feature_sets(meta, events)
    n_train = len(train)
    a, b = np.arange(n_train), np.arange(n_train, len(meta))
    y = train.target.to_numpy()
    general = ensemble_predict(frames, y, a, b)
    # Verify cached-feature inference also exactly reproduces the accepted general model.
    baseline_bytes = pd.DataFrame({"cookie_id": test.cookie_id, "score": general}).to_csv(
        index=False, lineterminator="\r\n").encode("utf-8")
    manifest = json.loads((Path(__file__).parent / "artifacts/manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(baseline_bytes).hexdigest() == manifest["submission_sha256"]
    specialist = specialist_predict(frames["pointer"], y, a, b)
    score = .85 * general + .15 * specialist
    submission = pd.DataFrame({"cookie_id": test.cookie_id, "score": score})
    assert submission.cookie_id.equals(test.cookie_id) and submission.cookie_id.is_unique
    assert np.isfinite(score).all() and submission.score.between(0, 1).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, lineterminator="\r\n")
    print("Saved candidate:", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()

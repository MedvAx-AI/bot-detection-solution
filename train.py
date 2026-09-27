"""Run inference or chronological validation from the original challenge files."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bot_detection.model import feature_sets, ensemble_predict, validate


def main():
    parser = argparse.ArgumentParser(description="Local cookie-level bot classification")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--validate", action="store_true", help="Run chronological validation instead of test inference")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data / "train.csv")
    test = pd.read_csv(args.data / "test.csv")
    events = pd.read_csv(args.data / "events.csv.gz", parse_dates=["event_ts"])
    meta = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    assert meta.cookie_id.is_unique and train.target.isin([0, 1]).all()
    frames = feature_sets(meta, events)
    if args.validate:
        validate(frames, train, args.output)
        return
    n = len(train)
    score = ensemble_predict(frames, train.target.to_numpy(), np.arange(n), np.arange(n, len(meta)))
    submission = pd.DataFrame({"cookie_id": test.cookie_id, "score": score})
    assert list(submission.columns) == ["cookie_id", "score"]
    assert len(submission) == len(test) and submission.cookie_id.is_unique
    assert submission.cookie_id.eq(test.cookie_id).all()
    assert np.isfinite(score).all() and submission.score.between(0, 1).all()
    submission.to_csv(args.output, index=False, lineterminator="\r\n")
    print("Saved", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()

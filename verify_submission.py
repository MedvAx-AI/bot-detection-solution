"""Check submission format and exact match against the recorded artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
args = parser.parse_args()
submission = pd.read_csv(args.submission)
test = pd.read_csv(args.test)
assert submission.columns.tolist() == ["cookie_id", "score"]
assert len(submission) == len(test) and submission.cookie_id.is_unique
assert submission.cookie_id.equals(test.cookie_id)
assert np.isfinite(submission.score).all() and submission.score.between(0, 1).all()
manifest = json.loads((Path(__file__).parent / "artifacts/manifest.json").read_text(encoding="utf-8"))
actual = hashlib.sha256(args.submission.read_bytes()).hexdigest()
assert actual == manifest["submission_sha256"], "Format is valid, but the output differs from the recorded submission"
print(f"OK: {len(submission)} cookies; exact SHA-256 match: {actual}")

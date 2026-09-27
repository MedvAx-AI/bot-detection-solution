"""Feature groups, deterministic training, and chronological validation."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from .features import build_features, pointer_features, device_features
from .metric import precision_at_recall


def specialist_predict(frame, y, train_index, predict_index):
    """Fit independent web/mobile models; the platform is an observed feature."""
    is_mobile = frame.mode_platform_norm.ne("web").to_numpy()
    categorical = frame.select_dtypes("category").columns.tolist()
    prediction = np.empty(len(predict_index))
    for mobile_group in [False, True]:
        group_train = train_index[is_mobile[train_index] == mobile_group]
        group_predict = is_mobile[predict_index] == mobile_group
        model = CatBoostClassifier(
            iterations=500, depth=5, learning_rate=.045, l2_leaf_reg=8,
            loss_function="Logloss", random_seed=42, verbose=False,
            thread_count=6, allow_writing_files=False,
        )
        model.fit(frame.iloc[group_train], y[group_train], cat_features=categorical)
        prediction[group_predict] = model.predict_proba(frame.iloc[predict_index[group_predict]])[:, 1]
        print("Trained specialist:", "mobile" if mobile_group else "web", flush=True)
    return prediction

# name, number of trees used at inference, ensemble weight, CPU threads
CONFIG = [
    ("base", 700, 0.125, 8),
    ("pointer", 500, 0.375, 8),
    ("deduplicated", 500, 0.250, 8),
    ("device", 500, 0.250, 6),
]


def feature_sets(meta, events):
    base = build_features(meta, events)
    pointer = base.merge(pointer_features(meta, events), on="cookie_id", validate="one_to_one")
    device = pointer.merge(device_features(meta, events), on="cookie_id", validate="one_to_one")
    print("Calculated features from recorded events", flush=True)
    # Remove only rows identical in every original event column.
    unique_events = events.drop_duplicates()
    clean = build_features(meta, unique_events).merge(
        pointer_features(meta, unique_events), on="cookie_id", validate="one_to_one")
    print("Calculated features after exact duplicate removal", flush=True)
    frames = {"base": base, "pointer": pointer, "deduplicated": clean, "device": device}
    for name, frame in frames.items():
        assert frame.cookie_id.reset_index(drop=True).equals(meta.cookie_id.reset_index(drop=True))
        x = frame.drop(columns="cookie_id").copy()
        for c in x.select_dtypes(include=["object", "str", "string"]).columns:
            x[c] = x[c].astype("category")
        frames[name] = x
    return frames


def ensemble_predict(frames, y, train_index, predict_index):
    result = np.zeros(len(predict_index), dtype=float)
    for name, trees, weight, threads in CONFIG:
        x = frames[name]
        categorical = x.select_dtypes(include="category").columns.tolist()
        clf = CatBoostClassifier(
            iterations=700, depth=6, learning_rate=0.045, l2_leaf_reg=5,
            loss_function="Logloss", random_seed=42, verbose=False,
            thread_count=threads, allow_writing_files=False,
        )
        clf.fit(x.iloc[train_index], y[train_index], cat_features=categorical)
        result += weight * clf.predict_proba(x.iloc[predict_index], ntree_end=trees)[:, 1]
        print("Trained", name, flush=True)
    return result


def validate(frames, train, output):
    dates = pd.to_datetime(train.window_start_ts)
    folds = [("early", "2026-04-11", "2026-04-14"),
             ("middle", "2026-04-14", "2026-04-17"),
             ("late", "2026-04-17", "2026-04-20")]
    rows = []
    y = train.target.to_numpy()
    for name, start, end in folds:
        a = np.flatnonzero(dates.lt(start))
        b = np.flatnonzero(dates.ge(start) & dates.lt(end))
        baseline_x = frames["base"][["n_events", "n_items"]]
        baseline = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=8)
        baseline.fit(baseline_x.iloc[a], y[a])
        baseline_score = baseline.predict_proba(baseline_x.iloc[b])[:, 1]
        score = ensemble_predict(frames, y, a, b)
        row = {"period": name, "validation_start": start, "validation_end_exclusive": end,
               "train_cookies": len(a), "validation_cookies": len(b),
               "positive_cookies": int(y[b].sum()),
               "baseline_precision_at_recall_0_70": precision_at_recall(y[b], baseline_score),
               "precision_at_recall_0_70": precision_at_recall(y[b], score),
               "average_precision": average_precision_score(y[b], score)}
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(output, index=False)

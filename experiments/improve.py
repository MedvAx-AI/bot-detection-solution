"""Compare alternate models on three forward-in-time folds.

Run from the repository root: python experiments/improve.py
Caches are local, tied to input hashes, and rebuilt when missing.
"""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bot_detection.model import feature_sets, ensemble_predict
from bot_detection.metric import precision_at_recall

DATA = ROOT / 'data'
WORK = ROOT / 'work/experiments'
WORK.mkdir(parents=True, exist_ok=True)
fingerprint = {n: hashlib.sha256((DATA/n).read_bytes()).hexdigest()
               for n in ['train.csv', 'test.csv', 'events.csv.gz']}
train = pd.read_csv(DATA/'train.csv')
test = pd.read_csv(DATA/'test.csv')
cache = WORK/'features.pkl'
cached = pickle.loads(cache.read_bytes()) if cache.exists() else None
if cached is not None and cached['fingerprint'] == fingerprint:
    frames = cached['frames']
else:
    meta = pd.concat([train.drop(columns='target'), test], ignore_index=True)
    events = pd.read_csv(DATA/'events.csv.gz', parse_dates=['event_ts'])
    frames = feature_sets(meta, events)
    cache.write_bytes(pickle.dumps({'fingerprint': fingerprint, 'frames': frames}))

x = frames['pointer'].iloc[:len(train)]
y = train.target.to_numpy()
dates = pd.to_datetime(train.window_start_ts)
mobile = x.mode_platform_norm.ne('web').to_numpy()
rows = []
folds = [('early','2026-04-11','2026-04-14'),
         ('middle','2026-04-14','2026-04-17'), ('late','2026-04-17','2026-04-20')]

for label,start,end in folds:
    a = np.flatnonzero(dates.lt(start))
    b = np.flatnonzero(dates.ge(start) & dates.lt(end))
    path = WORK/f'reference_{label}.pkl'
    reference_cache = pickle.loads(path.read_bytes()) if path.exists() else None
    if reference_cache is not None and reference_cache['fingerprint'] == fingerprint:
        assert np.array_equal(reference_cache['ids'], train.cookie_id.iloc[b].to_numpy())
        reference = reference_cache['score']
    else:
        reference = ensemble_predict(frames, y, a, b)
        path.write_bytes(pickle.dumps({'fingerprint': fingerprint, 'ids': train.cookie_id.iloc[b].to_numpy(), 'score': reference}))
    print(label, 'reference', precision_at_recall(y[b], reference), flush=True)
    for name in ['xgb3', 'xgb5', 'specialists', 'recent']:
        output = WORK/f'{name}_{label}.npy'
        if name.startswith('xgb'):
            model = XGBClassifier(n_estimators=1000, max_depth=int(name[-1]), learning_rate=.025,
                min_child_weight=3, reg_lambda=10, reg_alpha=.1, subsample=.85,
                colsample_bytree=.85, tree_method='hist', enable_categorical=True,
                max_cat_to_onehot=8, objective='binary:logistic', random_state=42, n_jobs=6)
            model.fit(x.iloc[a], y[a])
            prediction = model.predict_proba(x.iloc[b])[:,1]
        elif name == 'specialists':
            prediction = np.empty(len(b))
            for group in [False, True]:
                aa = a[mobile[a] == group]
                mask = mobile[b] == group
                model = CatBoostClassifier(iterations=500,depth=5,learning_rate=.045,l2_leaf_reg=8,
                    loss_function='Logloss',random_seed=42,verbose=False,thread_count=6,allow_writing_files=False)
                model.fit(x.iloc[aa], y[aa], cat_features=x.select_dtypes('category').columns.tolist())
                prediction[mask] = model.predict_proba(x.iloc[b[mask]])[:,1]
        else:
            age = (dates.iloc[a].max() - dates.iloc[a]).dt.total_seconds().to_numpy()/86400
            weights = np.exp2(-age/7)
            weights /= weights.mean()
            model = CatBoostClassifier(iterations=500,depth=6,learning_rate=.045,l2_leaf_reg=5,
                loss_function='Logloss',random_seed=42,verbose=False,thread_count=6,allow_writing_files=False)
            model.fit(x.iloc[a], y[a], sample_weight=weights, cat_features=x.select_dtypes('category').columns.tolist())
            prediction = model.predict_proba(x.iloc[b])[:,1]
        np.save(output, prediction)
        for weight in [0., .15, .3, .5, 1.]:
            blended = (1-weight)*reference + weight*prediction
            row = {'period':label,'model':name,'weight':weight,
                   'precision_at_recall':precision_at_recall(y[b],blended),
                   'average_precision':average_precision_score(y[b],blended)}
            rows.append(row)
            print(row,flush=True)
        pd.DataFrame(rows).to_csv(WORK/'results.csv',index=False)

summary = pd.DataFrame(rows).groupby(['model','weight'])[['precision_at_recall','average_precision']].mean()
print(summary.sort_values('precision_at_recall',ascending=False).to_string(),flush=True)

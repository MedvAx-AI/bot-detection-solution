from pathlib import Path
import sys,pickle,json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
repo=ROOT;sys.path.insert(0,str(repo))
sys.path.insert(0,str(repo/'experiments'))
from query_features import query_features
from bot_detection.metric import precision_at_recall
folder=repo/'work/experiments'
train=pd.read_csv(repo/'data/train.csv');test=pd.read_csv(repo/'data/test.csv')
meta=pd.concat([train.drop(columns='target'),test],ignore_index=True)
events=pd.read_csv(repo/'data/events.csv.gz',parse_dates=['event_ts'])
extra=query_features(meta,events)
extra.to_pickle(ROOT/'work/query_features.pkl')
frames=pickle.loads((folder/'features.pkl').read_bytes())['frames']
assert extra.cookie_id.equals(meta.cookie_id)
x=frames['pointer'].join(extra.drop(columns='cookie_id'))
x.mode_search_query=x.mode_search_query.astype('category')
y=train.target.to_numpy();dates=pd.to_datetime(train.window_start_ts)
rows=[]
for name,start,end in [('early','2026-04-11','2026-04-14'),('middle','2026-04-14','2026-04-17'),('late','2026-04-17','2026-04-20')]:
    a=np.flatnonzero(dates.lt(start));b=np.flatnonzero(dates.ge(start)&dates.lt(end))
    reference=pickle.loads((folder/f'reference_{name}.pkl').read_bytes())['score']
    model=CatBoostClassifier(iterations=700,depth=6,learning_rate=.045,l2_leaf_reg=5,
        loss_function='Logloss',random_seed=42,verbose=False,thread_count=6,allow_writing_files=False)
    model.fit(x.iloc[a],y[a],cat_features=x.select_dtypes('category').columns.tolist())
    for end_tree in [500,700]:
        pred=model.predict_proba(x.iloc[b],ntree_end=end_tree)[:,1]
        np.save(folder/f'query{end_tree}_{name}.npy',pred)
        for w in [0.,.15,.3,.5,1.]:
            score=(1-w)*reference+w*pred
            row={'period':name,'trees':end_tree,'weight':w,'precision_at_recall':precision_at_recall(y[b],score),
                 'average_precision':average_precision_score(y[b],score)}
            rows.append(row);print(row,flush=True)
    pd.DataFrame(rows).to_csv(folder/'query_results.csv',index=False)

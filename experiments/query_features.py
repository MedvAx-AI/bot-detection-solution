"""Within-window lexical properties of search queries; no external text models."""
import numpy as np
import pandas as pd


def query_features(meta, events):
    windows=meta[['cookie_id','window_start_ts','window_end_ts']].copy()
    for col in ['window_start_ts','window_end_ts']:
        windows[col]=pd.to_datetime(windows[col])
    events=events.merge(windows,on='cookie_id',validate='many_to_one')
    queries=events.loc[events.event_ts.ge(events.window_start_ts)
                      & events.event_ts.lt(events.window_end_ts)
                      & events.search_query.notna()].copy()
    text=queries.search_query.str.lower().str.strip()
    queries['normalized_query']=text
    queries['query_words']=text.str.split().str.len()
    queries['query_digits']=text.str.count(r'\d')/text.str.len().clip(lower=1)
    queries['query_latin']=text.str.count('[a-z]')/text.str.len().clip(lower=1)
    queries['query_number']=text.str.contains(r'\d')
    group=queries.groupby('cookie_id')
    result=windows.set_index('cookie_id')[[]].copy()
    result['mode_search_query']=group.normalized_query.agg(lambda x:x.mode().iloc[0])
    result['query_words_mean']=group.query_words.mean()
    result['query_words_max']=group.query_words.max()
    result['query_digits_mean']=group.query_digits.mean()
    result['query_latin_mean']=group.query_latin.mean()
    result['query_number_share']=group.query_number.mean()
    result['mode_search_query']=result.mode_search_query.fillna('missing')
    return result.replace([np.inf,-np.inf],np.nan).fillna(0).reset_index()

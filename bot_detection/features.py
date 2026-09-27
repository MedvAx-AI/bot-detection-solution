"""Window-limited activity, diversity, device and pointer features."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

EVENT_NAMES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login", "captcha_shown",
]


def entropy(series: pd.Series) -> float:
    p = series.value_counts(normalize=True, dropna=True)
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def mode(series: pd.Series) -> str:
    x = series.dropna()
    return str(x.mode().iloc[0]) if len(x) else "missing"


def build_features(meta: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    m["cookie_created_at"] = pd.to_datetime(m["cookie_created_at"])
    m["window_start_ts"] = pd.to_datetime(m["window_start_ts"])
    m["window_end_ts"] = pd.to_datetime(m["window_end_ts"])

    e = events.merge(m[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id", how="inner", validate="many_to_one")
    e = e.loc[(e.event_ts >= e.window_start_ts) & (e.event_ts < e.window_end_ts)].copy()
    e.sort_values(["cookie_id", "event_ts", "eid"], inplace=True)
    g = e.groupby("cookie_id", sort=False)

    f = m.set_index("cookie_id")[["window_start_ts", "window_end_ts", "cookie_created_at"]].copy()
    f["cookie_age_hours"] = (f.window_start_ts - f.cookie_created_at).dt.total_seconds() / 3600
    f["cookie_age_log"] = np.log1p(f.cookie_age_hours.clip(lower=0))
    f["created_hour"] = f.cookie_created_at.dt.hour
    f["created_weekday"] = f.cookie_created_at.dt.dayofweek
    f["window_weekday"] = f.window_start_ts.dt.dayofweek
    f["created_in_window"] = (f.cookie_created_at >= f.window_start_ts).astype(int)

    e["sec_in_window"] = (e.event_ts - e.window_start_ts).dt.total_seconds()
    e["hour"] = e.event_ts.dt.hour
    e["minute"] = e.event_ts.dt.hour * 60 + e.event_ts.dt.minute
    e["delta"] = g.event_ts.diff().dt.total_seconds()
    e["same_item_prev"] = e.item_id.eq(g.item_id.shift()) & e.item_id.notna()
    e["same_event_prev"] = e.eid.eq(g.eid.shift())
    e["same_category_prev"] = e.item_category.eq(g.item_category.shift()) & e.item_category.notna()
    e["platform_norm"] = e.platform.str.lower().replace({"desktop": "web", "iphone": "ios"})
    ua = e.user_agent.fillna("")
    e["browser"] = np.select(
        [ua.str.contains("YaBrowser", case=False), ua.str.contains("Firefox", case=False),
         ua.str.contains("Chrome", case=False), ua.str.contains("Safari", case=False)],
        ["yandex", "firefox", "chrome", "safari"], default="other")
    e["os"] = np.select(
        [ua.str.contains("Android", case=False), ua.str.contains("iPhone|iPad", case=False),
         ua.str.contains("Windows", case=False), ua.str.contains("Macintosh", case=False),
         ua.str.contains("Linux", case=False)],
        ["android", "ios", "windows", "mac", "linux"], default="other")

    core = g.agg(
        n_events=("eid", "size"), n_event_types=("eid", "nunique"),
        first_sec=("sec_in_window", "min"), last_sec=("sec_in_window", "max"),
        active_hours=("hour", "nunique"), active_minutes=("minute", "nunique"),
        n_items=("item_id", "nunique"), n_categories=("item_category", "nunique"),
        n_locations=("item_location", "nunique"), n_queries=("search_query", "nunique"),
        n_user_agents=("user_agent", "nunique"), n_platforms=("platform_norm", "nunique"),
        n_seller_types=("seller_type", "nunique"),
        item_filled=("item_id", "count"), pointer_filled=("pointer_x", "count"),
        query_filled=("search_query", "count"),
        same_item_count=("same_item_prev", "sum"),
        same_event_count=("same_event_prev", "sum"),
        same_category_count=("same_category_prev", "sum"),
        search_page_mean=("search_page", "mean"), search_page_max=("search_page", "max"),
        search_page_std=("search_page", "std"),
        pointer_x_mean=("pointer_x", "mean"), pointer_x_std=("pointer_x", "std"),
        pointer_y_mean=("pointer_y", "mean"), pointer_y_std=("pointer_y", "std"),
        pointer_x_unique=("pointer_x", "nunique"), pointer_y_unique=("pointer_y", "nunique"),
    )
    f = f.join(core)
    f["span_hours"] = (f.last_sec - f.first_sec) / 3600
    f["first_hour"] = f.first_sec / 3600
    f["last_hour"] = f.last_sec / 3600
    f["events_per_active_hour"] = f.n_events / (f.span_hours + 0.1)
    for a, b in [("n_items", "item_filled"), ("n_categories", "n_events"),
                 ("n_locations", "n_events"), ("n_queries", "query_filled"),
                 ("pointer_x_unique", "pointer_filled"), ("active_minutes", "n_events"),
                 ("same_item_count", "n_events"), ("same_event_count", "n_events"),
                 ("same_category_count", "n_events")]:
        f[f"ratio_{a}"] = f[a] / f[b].clip(lower=1)

    counts = pd.crosstab(e.cookie_id, e.event_name).reindex(columns=EVENT_NAMES, fill_value=0)
    counts.columns = ["cnt_" + c for c in counts.columns]
    f = f.join(counts)
    for col in counts:
        f["share_" + col[4:]] = f[col] / f.n_events.clip(lower=1)
    f["views_per_search"] = f.cnt_item_view / (f.cnt_search_results_view + 1)
    f["contacts_per_view"] = (f.cnt_contact_phone_show + f.cnt_contact_chat_open + f.cnt_contact_message_sent) / (f.cnt_item_view + 1)
    f["actions_per_view"] = (f.cnt_photo_swipe + f.cnt_favorite_add + f.cnt_seller_page_view) / (f.cnt_item_view + 1)

    deltas = g.delta.agg(["mean", "median", "std", "min", "max"])
    deltas.columns = ["delta_" + c for c in deltas]
    f = f.join(deltas)
    for threshold in [0, 1, 3, 10, 60, 300, 1800]:
        f[f"delta_le_{threshold}"] = e.delta.le(threshold).groupby(e.cookie_id).mean()
    f["delta_gt_1800"] = e.delta.gt(1800).groupby(e.cookie_id).sum()
    f["delta_cv"] = f.delta_std / (f.delta_mean + 1)
    f["delta_p10"] = g.delta.quantile(0.1)
    f["delta_p90"] = g.delta.quantile(0.9)

    for col in ["hour", "item_id", "item_category", "item_location", "search_query", "eid"]:
        f["entropy_" + col] = g[col].apply(entropy)
    for col in ["platform_norm", "browser", "os", "user_agent", "item_category", "item_location", "seller_type"]:
        f["mode_" + col] = g[col].agg(mode)
    for col in ["platform_norm", "browser", "os", "seller_type"]:
        tab = pd.crosstab(e.cookie_id, e[col])
        for value in tab.columns:
            f[f"share_{col}_{value}"] = tab[value] / f.n_events.clip(lower=1)

    # Behaviour in the beginning and end of the day, independent of event count.
    for lo, hi, label in [(0, 6, "night"), (6, 12, "morning"), (12, 18, "day"), (18, 24, "evening")]:
        f[f"share_{label}"] = e.hour.between(lo, hi - 1).groupby(e.cookie_id).mean()
    e["query_len"] = e.search_query.str.len()
    f["query_len_mean"] = g.query_len.mean()
    f["query_len_std"] = g.query_len.std()
    f["query_len_max"] = g.query_len.max()

    # Per-action diversity and timing reveal repeated scraping flows.
    for name in ["search_results_view", "item_view", "photo_swipe", "seller_page_view", "contact_phone_show", "favorite_add", "captcha_shown"]:
        z = e.loc[e.event_name.eq(name)]
        zg = z.groupby("cookie_id")
        f[f"{name}_first_hour"] = zg.sec_in_window.min() / 3600
        f[f"{name}_last_hour"] = zg.sec_in_window.max() / 3600
        if name in ["item_view", "photo_swipe", "seller_page_view", "contact_phone_show", "favorite_add"]:
            f[f"{name}_unique_items"] = zg.item_id.nunique()
            f[f"{name}_unique_categories"] = zg.item_category.nunique()
            f[f"{name}_unique_locations"] = zg.item_location.nunique()
            f[f"{name}_items_per_event"] = f[f"{name}_unique_items"] / f[f"cnt_{name}"].clip(lower=1)

    f.drop(columns=["window_start_ts", "window_end_ts", "cookie_created_at"], inplace=True)
    f = f.replace([np.inf, -np.inf], np.nan)
    cats = f.select_dtypes(include=["object", "str", "string"]).columns.tolist()
    f[cats] = f[cats].fillna("missing").astype(str)
    f = f.fillna({c: 0 for c in f.columns if c not in cats})
    return f.reset_index()


def pointer_features(meta: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    m = meta[["cookie_id", "window_start_ts", "window_end_ts"]].copy()
    m["window_start_ts"] = pd.to_datetime(m.window_start_ts)
    m["window_end_ts"] = pd.to_datetime(m.window_end_ts)
    e = events.merge(m, on="cookie_id", how="inner", validate="many_to_one")
    e = e.loc[e.event_ts.ge(e.window_start_ts) & e.event_ts.lt(e.window_end_ts)].copy()
    e.sort_values(["cookie_id", "event_ts", "eid"], inplace=True)
    p = e.loc[e.pointer_x.notna() & e.pointer_y.notna()].copy()
    g = p.groupby("cookie_id", sort=False)
    p["dx"] = g.pointer_x.diff().abs()
    p["dy"] = g.pointer_y.diff().abs()
    p["distance"] = np.hypot(p.dx, p.dy)
    p["repeat_pointer"] = p.distance.eq(0)
    p["move_small"] = p.distance.le(10)
    g = p.groupby("cookie_id", sort=False)

    out = m.set_index("cookie_id")[[]].copy()
    out["pointer_dx_mean"] = g.dx.mean()
    out["pointer_dist_mean"] = g.distance.mean()
    out["pointer_dist_max"] = g.distance.max()
    out["pointer_x_range"] = g.pointer_x.max() - g.pointer_x.min()
    out["pointer_x_max_extra"] = g.pointer_x.max()
    out["pointer_y_max_extra"] = g.pointer_y.max()
    out["pointer_dist_median"] = g.distance.median()
    out["pointer_y_range"] = g.pointer_y.max() - g.pointer_y.min()
    out["pointer_repeat_rate"] = g.repeat_pointer.mean()
    out["pointer_small_move_rate"] = g.move_small.mean()
    return out.fillna(0).reset_index()


def device_features(meta: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Availability of pointer data and shape of observed pointer movements."""
    m = meta[["cookie_id", "window_start_ts", "window_end_ts"]].copy()
    for c in ["window_start_ts", "window_end_ts"]:
        m[c] = pd.to_datetime(m[c])
    e = events.merge(m, on="cookie_id", validate="many_to_one")
    e = e[e.event_ts.ge(e.window_start_ts) & e.event_ts.lt(e.window_end_ts)].copy()
    e.sort_values(["cookie_id", "event_ts", "eid"], inplace=True)
    f = m.set_index("cookie_id")[[]].copy()
    ua = e.user_agent.fillna("")
    e["headless"] = ua.str.contains("Headless", case=False)
    e["script"] = ua.str.contains("python|Scrapy|curl|node-fetch|Go-http", case=False)
    e["native"] = ua.str.startswith("Avito/")
    e["pointer"] = e.pointer_x.notna() & e.pointer_y.notna()
    for c in ["headless", "script", "native", "pointer"]:
        f[c + "_fraction"] = e.groupby("cookie_id")[c].mean()
    for name in ["search_results_view", "item_view", "photo_swipe", "contact_phone_show"]:
        f["pointer_fraction_" + name] = e[e.event_name.eq(name)].groupby("cookie_id").pointer.mean()

    p = e[e.pointer].copy()
    g = p.groupby("cookie_id", sort=False)
    p["dx_signed"] = g.pointer_x.diff()
    p["dy_signed"] = g.pointer_y.diff()
    p["distance"] = np.hypot(p.dx_signed, p.dy_signed)
    p["xy_product"] = p.pointer_x * p.pointer_y
    p["coordinate"] = p.pointer_x.astype(str) + "/" + p.pointer_y.astype(str)
    # Count direction changes only when both current and preceding moves exist.
    p["turn_x"] = (np.sign(p.dx_signed).ne(np.sign(g.dx_signed.shift()))
                   & p.dx_signed.notna() & g.dx_signed.shift().notna())
    p["turn_y"] = (np.sign(p.dy_signed).ne(np.sign(g.dy_signed.shift()))
                   & p.dy_signed.notna() & g.dy_signed.shift().notna())
    p["corner"] = p.pointer_x.le(100) & p.pointer_y.le(100)
    g = p.groupby("cookie_id", sort=False)
    for c in ["turn_x", "turn_y", "corner"]:
        f["ptr_" + c] = g[c].mean()
    f["ptr_unique_ratio"] = g.coordinate.nunique() / g.size()
    f["ptr_xy_corr"] = (
        (g.xy_product.mean() - g.pointer_x.mean() * g.pointer_y.mean())
        / (g.pointer_x.std(ddof=0) * g.pointer_y.std(ddof=0) + 1)
    )
    f["ptr_step_cv"] = g.distance.std() / (g.distance.mean() + 1)
    f["ptr_path_efficiency"] = np.hypot(
        g.pointer_x.last() - g.pointer_x.first(),
        g.pointer_y.last() - g.pointer_y.first(),
    ) / (g.distance.sum() + 1)
    for c in ["pointer_x", "pointer_y"]:
        f[c + "_median_device"] = g[c].median()
        f[c + "_min_device"] = g[c].min()
        f[c + "_skew_device"] = g[c].skew()
        f[c + "_iqr_device"] = g[c].quantile(.75) - g[c].quantile(.25)
    return f.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index()

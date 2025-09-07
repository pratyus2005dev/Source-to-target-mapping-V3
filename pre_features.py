# features.py
from typing import Dict
import pandas as pd
import numpy as np
import random
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils.io_utils import parse_ddl, load_csvs
from .profiler import profile_all

# -----------------------------
# STRING SIMILARITY FUNCTIONS
# -----------------------------
def levenshtein_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def jaccard_similarity(s1: str, s2: str) -> float:
    set1, set2 = set(s1.lower()), set(s2.lower())
    return len(set1 & set2) / max(1, len(set1 | set2))

def cosine_sim_str(s1: str, s2: str) -> float:
    vect = TfidfVectorizer(analyzer='char', ngram_range=(2,3))
    tfidf = vect.fit_transform([s1, s2])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0,0])

def normalize_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())

def string_similarity(s1: str, s2: str) -> float:
    return np.mean([
        levenshtein_ratio(s1, s2),
        jaccard_similarity(s1, s2),
        cosine_sim_str(s1, s2)
    ])

# -----------------------------
# DATA TYPE & LENGTH HELPERS
# -----------------------------
def _dtype_family(dtype: str) -> str:
    if not dtype:
        return "other"
    d = dtype.lower()
    if any(k in d for k in ["int", "decimal", "numeric", "float", "double"]):
        return "number"
    if any(k in d for k in ["char", "varchar", "text", "string"]):
        return "text"
    if "date" in d or "time" in d:
        return "date"
    return "other"

def _len_from_dtype(dtype: str) -> int:
    if not dtype:
        return 0
    m = re.search(r"\((\d+)", dtype)
    return int(m.group(1)) if m else 0

# -----------------------------
# EXTRA DATA FEATURES
# -----------------------------
def _extra_data_features(s_vals: pd.Series, t_vals: pd.Series) -> Dict[str, float]:
    feats = {
        "value_overlap": 0.0,
        "mean_diff": 0.0,
        "std_diff": 0.0,
        "avg_strlen_diff": 0.0
    }
    try:
        s_vals, t_vals = s_vals.dropna(), t_vals.dropna()
        if s_vals.empty or t_vals.empty:
            return feats

        s_sample = s_vals.sample(min(200, len(s_vals)), random_state=42)
        t_sample = t_vals.sample(min(200, len(t_vals)), random_state=42)

        # Value overlap
        s_set, t_set = set(map(str, s_sample)), set(map(str, t_sample))
        feats["value_overlap"] = len(s_set & t_set) / max(1, len(s_set | t_set))

        # Numeric stats
        if pd.api.types.is_numeric_dtype(s_sample) and pd.api.types.is_numeric_dtype(t_sample):
            feats["mean_diff"] = abs(s_sample.mean() - t_sample.mean())
            feats["std_diff"] = abs(s_sample.std() - t_sample.std())

        # String length difference
        feats["avg_strlen_diff"] = abs(s_sample.astype(str).str.len().mean() - t_sample.astype(str).str.len().mean())
    except Exception:
        pass
    return feats

# -----------------------------
# PAIRWISE FEATURE EXTRACTION
# -----------------------------
def cosine_ngrams(s1: str, s2: str, n: int = 3) -> float:
    s1, s2 = f"^{s1}$", f"^{s2}$"
    a = [s1[i:i+n] for i in range(max(1, len(s1)-n+1))]
    b = [s2[i:i+n] for i in range(max(1, len(s2)-n+1))]
    if not a or not b:
        return 0.0
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    num = sum(ca[k]*cb.get(k,0) for k in ca)
    da = np.sqrt(sum(v*v for v in ca.values()))
    db = np.sqrt(sum(v*v for v in cb.values()))
    return num/(da*db) if da and db else 0.0

def _pair_features(s_tbl, s_col, t_tbl, t_col, ddl_map, prof_src, prof_tgt, s_df=None, t_df=None):
    fuzzy = string_similarity(s_col, t_col)
    norm_eq = float(normalize_name(s_col) == normalize_name(t_col))
    lev_ratio = levenshtein_ratio(s_col, t_col)
    jacc = jaccard_similarity(s_col, t_col)
    cosine = cosine_ngrams(s_col, t_col, n=3)

    # DDL meta
    s_dt = ddl_map.get(s_tbl, {}).get(s_col)
    t_dt = ddl_map.get(t_tbl, {}).get(t_col)
    type_match = 1.0 if s_dt and t_dt and _dtype_family(s_dt) == _dtype_family(t_dt) else 0.0
    len_diff = abs(_len_from_dtype(s_dt) - _len_from_dtype(t_dt))

    # Profile meta
    sprof = prof_src.get(s_tbl, {}).get(s_col, {})
    tprof = prof_tgt.get(t_tbl, {}).get(t_col, {})
    null_gap = abs((sprof.get("null_ratio",0.0) or 0.0) - (tprof.get("null_ratio",0.0) or 0.0))
    uniq_gap = abs((sprof.get("distinct_count",0) or 0) - (tprof.get("distinct_count",0) or 0))
    s_top, t_top = set(sprof.get("top_values",[])), set(tprof.get("top_values",[]))
    top_overlap = float(len(s_top & t_top) / max(1, len(s_top | t_top))) if (s_top or t_top) else 0.0

    extra_feats = _extra_data_features(s_df[s_col], t_df[t_col]) if s_df is not None and t_df is not None and s_col in s_df.columns and t_col in t_df.columns else {
        "value_overlap":0.0, "mean_diff":0.0, "std_diff":0.0, "avg_strlen_diff":0.0
    }

    return {
        "fuzzy_name": float(fuzzy),
        "name_exact_norm": float(norm_eq),
        "levenshtein": float(lev_ratio),
        "jaccard": float(jacc),
        "cosine": float(cosine),
        "dtype_compat": float(type_match),
        "len_diff": float(len_diff),
        "null_gap": float(null_gap),
        "uniq_gap": float(uniq_gap),
        "top_overlap": float(top_overlap),
        **extra_feats
    }

# -----------------------------
# BUILD FEATURE MATRIX
# -----------------------------
def build_feature_matrix(cfg, synonyms: Dict[str, list], sample_rows: int = 1000,
                         negative_ratio: float = None, save_csv: str = None):
    negative_ratio = getattr(cfg.train, "negative_ratio", 1.0) if negative_ratio is None else negative_ratio
    ddl_map = parse_ddl(getattr(cfg.paths, "ddl_path", ""))
    src_tables = load_csvs(cfg.paths.source_root, cfg.paths.source_files, nrows=sample_rows)
    tgt_tables = load_csvs(cfg.paths.target_root, cfg.paths.target_files, nrows=sample_rows)
    prof_src = profile_all(cfg.paths.source_root, cfg.paths.source_files, sample_rows=sample_rows)
    prof_tgt = profile_all(cfg.paths.target_root, cfg.paths.target_files, sample_rows=sample_rows)

    src_cols = [(tbl, col) for tbl, df in src_tables.items() if not df.empty for col in df.columns]
    tgt_cols = [(tbl, col) for tbl, df in tgt_tables.items() if not df.empty for col in df.columns]

    rows, labels = [], []

    # Positive examples
    for src_key, tgt_list in (synonyms or {}).items():
        try:
            s_tbl, s_col = src_key.split("::", 1)
        except:
            continue
        for tgt_key in tgt_list:
            try:
                t_tbl, t_col = tgt_key.split("::", 1)
            except:
                continue
            feats = _pair_features(s_tbl, s_col, t_tbl, t_col, ddl_map, prof_src, prof_tgt,
                                   src_tables.get(s_tbl), tgt_tables.get(t_tbl))
            rows.append({"source_table": s_tbl, "source_column": s_col,
                         "target_table": t_tbl, "target_column": t_col, **feats})
            labels.append(1)

    # Negative examples
    rng = random.Random(getattr(cfg.train, "random_state", 42))
    neg_needed = int(len(labels) * negative_ratio) if labels else max(100, len(src_cols) * len(tgt_cols)//50)
    for _ in range(neg_needed):
        if not src_cols or not tgt_cols:
            break
        s_tbl, s_col = src_cols[rng.randrange(len(src_cols))]
        t_tbl, t_col = tgt_cols[rng.randrange(len(tgt_cols))]
        src_key, tgt_key = f"{s_tbl}::{s_col}", f"{t_tbl}::{t_col}"
        if (src_key in (synonyms or {}) and tgt_key in (synonyms or {}).get(src_key, [])):
            continue
        feats = _pair_features(s_tbl, s_col, t_tbl, t_col, ddl_map, prof_src, prof_tgt,
                               src_tables.get(s_tbl), tgt_tables.get(t_tbl))
        rows.append({"source_table": s_tbl, "source_column": s_col,
                     "target_table": t_tbl, "target_column": t_col, **feats})
        labels.append(0)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    full = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")

    feature_cols = [
        "fuzzy_name", "name_exact_norm", "levenshtein", "jaccard", "cosine",
        "dtype_compat", "len_diff", "null_gap", "uniq_gap", "top_overlap",
        "value_overlap", "mean_diff", "std_diff", "avg_strlen_diff"
    ]

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in full.columns:
            full[col] = 0.0

    X = full[feature_cols].astype(float).fillna(0.0)

    if save_csv:
        full.to_csv(save_csv, index=False)
        print(f"[INFO] Pretraining features CSV saved at {save_csv}")

    return X, y, full

def build_features(cfg, synonyms, sample_rows: int = 1000):
    return build_feature_matrix(cfg, synonyms, sample_rows)

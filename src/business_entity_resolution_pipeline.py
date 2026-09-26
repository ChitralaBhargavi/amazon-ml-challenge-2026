"""Business Entity Resolution pipeline for the Amazon ML Challenge 2026.

This version keeps the real challenge data and uses a practical blocking +
feature-based model design that is fast enough to run on a single workstation.
It is optimized to avoid unnecessary in-memory copies of the large TSV files and to
stream the heaviest source data through the candidate-index construction.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", "pvt",
    "private", "plc", "co", "company", "companies", "group", "grp", "llp", "sarl",
    "sas", "sa", "service", "services", "solutions", "consultants", "consulting",
    "trading", "foundation", "enterprise", "enterprises", "industries",
}

STOPWORDS = {
    "and", "the", "of", "for", "on", "in", "at", "to", "from", "near", "new",
    "old", "east", "west", "north", "south", "center", "centre", "road", "street",
    "st", "rd", "ave", "avenue", "suite", "unit", "blk", "lane", "ln", "building",
    "apartment", "apt",
}


@dataclass(slots=True)
class Record:
    entity_id: str
    country: str
    name_tokens: Tuple[str, ...]
    address_tokens: Tuple[str, ...]
    normalized_name: str
    normalized_address: str


def log_stage(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def read_tsv(path: Path, *, nrows: Optional[int] = None, sample_n: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    dtypes = {
        "entity_id": "string",
        "business_name": "string",
        "business_address": "string",
        "country": "string",
    }
    if path.name.endswith("ground_truth.tsv"):
        dtypes = {"source1_entity_id": "string", "matched_entity_ids": "string"}

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=dtypes,
        keep_default_na=False,
        na_filter=False,
        nrows=nrows,
        low_memory=False,
    )
    if sample_n is not None and 0 < sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    return df


def iter_source_records(path: Path, *, chunk_size: int = 200_000) -> Iterator[Record]:
    dtypes = {
        "entity_id": "string",
        "business_name": "string",
        "business_address": "string",
        "country": "string",
    }

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=dtypes,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
        low_memory=False,
    ):
        for row in chunk.itertuples(index=False, name=None):
            yield to_record_values(row[0], row[1], row[2], row[3])


def ascii_normalize(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    return text.encode("ascii", "ignore").decode("ascii")


def clean_text(raw: Optional[str], *, is_name: bool = False) -> str:
    text = ascii_normalize(raw or "")
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("+", " ")
    text = re.sub(r"[/_\\|~]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if is_name:
        for suffix in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(suffix)}\b", " ", text)
    return text


def tokenise(raw: Optional[str], *, drop_stopwords: bool = False) -> Tuple[str, ...]:
    cleaned = clean_text(raw)
    if not cleaned:
        return ()
    tokens = [t for t in cleaned.split() if t]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tuple(t for t in tokens if len(t) > 1)


def to_record_values(entity_id: object, business_name: object, business_address: object, country: object) -> Record:
    name_raw = str(business_name or "")
    address_raw = str(business_address or "")
    name_clean = clean_text(name_raw, is_name=True)
    address_clean = clean_text(address_raw)
    return Record(
        entity_id=str(entity_id),
        country=str(country or ""),
        name_tokens=tokenise(name_clean, drop_stopwords=True),
        address_tokens=tokenise(address_clean),
        normalized_name=name_clean,
        normalized_address=address_clean,
    )


def to_record(row: pd.Series) -> Record:
    return to_record_values(
        row.get("entity_id"),
        row.get("business_name"),
        row.get("business_address"),
        row.get("country"),
    )


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_score(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def seq_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio()


def pair_features(r1: Record, r2: Record) -> Dict[str, float]:
    name_tokens_1 = r1.name_tokens
    name_tokens_2 = r2.name_tokens
    addr_tokens_1 = r1.address_tokens
    addr_tokens_2 = r2.address_tokens

    return {
        "same_country": 1.0 if r1.country and r1.country == r2.country else 0.0,
        "same_name_exact": 1.0 if r1.normalized_name and r1.normalized_name == r2.normalized_name else 0.0,
        "same_address_exact": 1.0 if r1.normalized_address and r1.normalized_address == r2.normalized_address else 0.0,
        "name_jaccard": jaccard(name_tokens_1, name_tokens_2),
        "address_jaccard": jaccard(addr_tokens_1, addr_tokens_2),
        "name_overlap": overlap_score(name_tokens_1, name_tokens_2),
        "address_overlap": overlap_score(addr_tokens_1, addr_tokens_2),
        "name_char_sim": seq_sim(r1.normalized_name, r2.normalized_name),
        "address_char_sim": seq_sim(r1.normalized_address, r2.normalized_address),
        "first_token_match": 1.0 if name_tokens_1 and name_tokens_2 and name_tokens_1[0] == name_tokens_2[0] else 0.0,
        "last_token_match": 1.0 if name_tokens_1 and name_tokens_2 and name_tokens_1[-1] == name_tokens_2[-1] else 0.0,
        "name_len_ratio": min(len(r1.normalized_name), len(r2.normalized_name)) / max(1, max(len(r1.normalized_name), len(r2.normalized_name))),
        "address_len_ratio": min(len(r1.normalized_address), len(r2.normalized_address)) / max(1, max(len(r1.normalized_address), len(r2.normalized_address))),
        "num_name_tokens_1": len(name_tokens_1),
        "num_name_tokens_2": len(name_tokens_2),
        "num_address_tokens_1": len(addr_tokens_1),
        "num_address_tokens_2": len(addr_tokens_2),
    }


def build_var_index(records: Sequence[Record]) -> Dict[Tuple[str, str], Set[str]]:
    index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for rec in records:
        if not rec.country:
            continue
        for token in rec.name_tokens + rec.address_tokens[:6]:
            if token:
                index[(rec.country, token)].add(rec.entity_id)
        if rec.normalized_name:
            index[(rec.country, rec.normalized_name)].add(rec.entity_id)
        if rec.name_tokens:
            bigram = " ".join(rec.name_tokens[:2])
            if bigram:
                index[(rec.country, bigram)].add(rec.entity_id)
    return index


def generate_candidates(
    s1: Record,
    candidate_records: Sequence[Record] | Dict[str, Record],
    index: Dict[Tuple[str, str], Set[str]],
    max_candidates: int = 8,
) -> List[str]:
    if not s1.country:
        return []

    candidate_ids: Set[str] = set()
    for token in s1.name_tokens[:6] + s1.address_tokens[:6]:
        if token:
            candidate_ids |= index.get((s1.country, token), set())
    if s1.normalized_name:
        candidate_ids |= index.get((s1.country, s1.normalized_name), set())
    if s1.name_tokens:
        bigram = " ".join(s1.name_tokens[:2])
        if bigram:
            candidate_ids |= index.get((s1.country, bigram), set())

    if isinstance(candidate_records, dict):
        rec_by_id = candidate_records
    else:
        rec_by_id = {rec.entity_id: rec for rec in candidate_records}

    scored: List[Tuple[float, str]] = []
    for cand_id in candidate_ids:
        cand = rec_by_id.get(cand_id)
        if cand is None or cand.country != s1.country:
            continue
        score = 0.0
        if s1.normalized_name and cand.normalized_name:
            score += 3.0 if s1.normalized_name == cand.normalized_name else 0.0
        score += 2.0 * jaccard(s1.name_tokens, cand.name_tokens)
        score += 1.0 * overlap_score(s1.address_tokens, cand.address_tokens)
        score += 0.5 * max(0.0, seq_sim(s1.normalized_name, cand.normalized_name))
        if s1.name_tokens and cand.name_tokens:
            if s1.name_tokens[0] == cand.name_tokens[0]:
                score += 1.0
            if s1.name_tokens[-1] == cand.name_tokens[-1]:
                score += 1.0
        scored.append((score, cand_id))

    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entity_id for _, entity_id in scored[:max_candidates]]


def build_truth_map(truth_df: pd.DataFrame) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for row in truth_df.itertuples(index=False, name=None):
        sid = str(row[0])
        ids = [x.strip() for x in str(row[1] or "").split(",") if x.strip()]
        out[sid] = set(ids)
    return out


def f05_entity(true_ids: Set[str], pred_ids: Set[str]) -> float:
    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def build_train_dataset(
    train_s1: pd.DataFrame,
    train_s2: pd.DataFrame,
    train_s3: pd.DataFrame,
    truth_df: pd.DataFrame,
    max_s1: Optional[int] = 2000,
    max_candidates: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    if max_s1 is not None and 0 < max_s1 < len(train_s1):
        train_s1 = train_s1.sample(n=max_s1, random_state=42).reset_index(drop=True)

    candidate_map: Dict[str, Record] = {}
    for frame in (train_s2, train_s3):
        for row in frame.itertuples(index=False, name=None):
            rec = to_record_values(row[0], row[1], row[2], row[3])
            candidate_map[rec.entity_id] = rec

    index = build_var_index(candidate_map.values())
    truth_map = build_truth_map(truth_df)

    rows: List[dict] = []
    candidate_ids_by_s1: Dict[str, List[str]] = {}
    for row in train_s1.itertuples(index=False, name=None):
        s1 = to_record_values(row[0], row[1], row[2], row[3])
        cand_ids = generate_candidates(s1, candidate_map, index, max_candidates=max_candidates)
        candidate_ids_by_s1[s1.entity_id] = cand_ids
        for cand_id in cand_ids:
            cand = candidate_map.get(cand_id)
            if cand is None:
                continue
            feat = pair_features(s1, cand)
            feat["source1_entity_id"] = s1.entity_id
            feat["candidate_entity_id"] = cand.entity_id
            feat["label"] = 1 if cand.entity_id in truth_map.get(s1.entity_id, set()) else 0
            rows.append(feat)

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise RuntimeError("No training pairs were generated from the blocker.")

    feature_cols = [c for c in pair_df.columns if c not in {"source1_entity_id", "candidate_entity_id", "label"}]
    s1_ids = train_s1["entity_id"].astype(str).tolist()
    _, valid_ids = train_test_split(s1_ids, test_size=0.15, random_state=42)

    valid_rows: List[dict] = []
    for sid in valid_ids:
        s1_row = train_s1.loc[train_s1["entity_id"].astype(str) == sid]
        if s1_row.empty:
            continue
        s1 = to_record(s1_row.iloc[0])
        for cand_id in candidate_ids_by_s1.get(sid, []):
            cand = candidate_map.get(cand_id)
            if cand is None:
                continue
            feat = pair_features(s1, cand)
            feat["source1_entity_id"] = sid
            feat["candidate_entity_id"] = cand.entity_id
            feat["label"] = 1 if cand.entity_id in truth_map.get(sid, set()) else 0
            valid_rows.append(feat)

    valid_df = pd.DataFrame(valid_rows)
    if valid_df.empty:
        valid_df = pd.DataFrame(columns=[*feature_cols, "source1_entity_id", "candidate_entity_id", "label"])

    return pair_df, valid_df, candidate_ids_by_s1


def select_threshold(model: HistGradientBoostingClassifier, valid_df: pd.DataFrame, truth_df: pd.DataFrame) -> Tuple[float, float]:
    if valid_df.empty:
        return 0.5, 0.0
    feature_cols = [c for c in valid_df.columns if c not in {"source1_entity_id", "candidate_entity_id", "label"}]
    scores = model.predict_proba(valid_df[feature_cols].to_numpy(dtype=float))[:, 1]
    valid_df = valid_df.copy()
    valid_df["score"] = scores
    truth_map = build_truth_map(truth_df)

    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        values = []
        for sid in sorted(valid_df["source1_entity_id"].unique()):
            pred_ids = set(valid_df.loc[(valid_df["source1_entity_id"] == sid) & (valid_df["score"] >= threshold), "candidate_entity_id"])
            values.append(f05_entity(truth_map.get(str(sid), set()), set(pred_ids)))
        metric = float(np.mean(values)) if values else 1.0
        if metric > best_score:
            best_score = metric
            best_threshold = float(threshold)
    return best_threshold, best_score


def generate_test_outputs(
    model: HistGradientBoostingClassifier,
    threshold: float,
    max_candidates: int = 6,
    *,
    test_s1: Optional[pd.DataFrame] = None,
    test_s2: Optional[pd.DataFrame] = None,
    test_s3: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_s1 is None:
        test_s1 = read_tsv(TEST_DIR / "test_source1.tsv")
    if test_s2 is None:
        test_s2 = read_tsv(TEST_DIR / "test_source2.tsv")
    if test_s3 is None:
        test_s3 = read_tsv(TEST_DIR / "test_source3.tsv")

    candidate_map: Dict[str, Record] = {}
    for frame in (test_s2, test_s3):
        for row in frame.itertuples(index=False, name=None):
            rec = to_record_values(row[0], row[1], row[2], row[3])
            candidate_map[rec.entity_id] = rec
    index = build_var_index(candidate_map.values())

    feature_cols = [
        "same_country", "same_name_exact", "same_address_exact", "name_jaccard", "address_jaccard",
        "name_overlap", "address_overlap", "name_char_sim", "address_char_sim",
        "first_token_match", "last_token_match", "name_len_ratio", "address_len_ratio",
        "num_name_tokens_1", "num_name_tokens_2", "num_address_tokens_1", "num_address_tokens_2",
    ]

    candidate_rows: List[dict] = []
    match_rows: List[dict] = []
    for row in test_s1.itertuples(index=False, name=None):
        s1 = to_record_values(row[0], row[1], row[2], row[3])
        cand_ids = generate_candidates(s1, candidate_map, index, max_candidates=max_candidates)
        candidate_rows.append({"source1_entity_id": s1.entity_id, "candidate_entity_ids": ",".join(cand_ids)})

        if not cand_ids:
            match_rows.append({"source1_entity_id": s1.entity_id, "matched_entity_ids": ""})
            continue

        hits = []
        for cand_id in cand_ids:
            cand = candidate_map.get(cand_id)
            if cand is None:
                continue
            feat = pair_features(s1, cand)
            feat_arr = np.asarray([[float(feat.get(col, 0.0)) for col in feature_cols]], dtype=float)
            score = float(model.predict_proba(feat_arr)[0, 1])
            if score >= threshold:
                hits.append(cand_id)
        match_rows.append({"source1_entity_id": s1.entity_id, "matched_entity_ids": ",".join(sorted(set(hits)))})

    return pd.DataFrame(candidate_rows), pd.DataFrame(match_rows)


def run_small_test() -> None:
    sample_s1 = read_tsv(TRAIN_DIR / "train_source1.tsv", sample_n=25, random_state=42)
    sample_s2 = read_tsv(TRAIN_DIR / "train_source2.tsv", sample_n=40, random_state=42)
    sample_s3 = read_tsv(TRAIN_DIR / "train_source3.tsv", sample_n=40, random_state=42)
    truth_df = read_tsv(TRAIN_DIR / "train_ground_truth.tsv")
    truth_df = truth_df[truth_df["source1_entity_id"].astype(str).isin(sample_s1["entity_id"].astype(str).tolist())].copy().reset_index(drop=True)

    print("=== Small-scale validation run ===")
    print(f"TSV loading: S1={len(sample_s1)}, S2={len(sample_s2)}, S3={len(sample_s3)}, ground truth={len(truth_df)}")

    pair_df, valid_df, _ = build_train_dataset(sample_s1, sample_s2, sample_s3, truth_df, max_s1=25, max_candidates=6)
    feature_cols = [c for c in pair_df.columns if c not in {"source1_entity_id", "candidate_entity_id", "label"}]
    print(f"Normalization/feature extraction: {len(feature_cols)} feature columns, {len(pair_df)} candidate pairs")

    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=50, random_state=42)
    model.fit(pair_df[feature_cols].to_numpy(dtype=float), pair_df["label"].to_numpy(dtype=int))
    threshold, macro_f05 = select_threshold(model, valid_df, truth_df)
    print(f"Model training + prediction + F0.5: threshold={threshold:.3f}, macro_F0.5={macro_f05:.4f}")

    test_s1 = read_tsv(TEST_DIR / "test_source1.tsv", sample_n=12, random_state=7)
    test_s2 = read_tsv(TEST_DIR / "test_source2.tsv", sample_n=18, random_state=7)
    test_s3 = read_tsv(TEST_DIR / "test_source3.tsv", sample_n=18, random_state=7)
    candidate_df, match_df = generate_test_outputs(model, threshold, max_candidates=6, test_s1=test_s1, test_s2=test_s2, test_s3=test_s3)

    sample_output_dir = OUTPUT_DIR / "small_test"
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(sample_output_dir / "candidate_pairs.tsv", sep="\t", index=False)
    match_df.to_csv(sample_output_dir / "matching_results.tsv", sep="\t", index=False)
    test_dir = sample_output_dir
    test_s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    test_s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    test_s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)
    print(f"Output generation: {len(candidate_df)} candidate rows, {len(match_df)} match rows")

    validator = [sys.executable, str(ROOT / "utils" / "validate_submission.py"), "--matching", str(test_dir / "matching_results.tsv"), "--candidate", str(test_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir)]
    result = subprocess.run(validator, capture_output=True, text=True, cwd=str(ROOT))
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    print(f"Submission validation exit code: {result.returncode}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("--train-max-s1", type=int, default=2000, help="Maximum S1 rows used for training; set 0 to use all rows.")
    parser.add_argument("--candidate-limit", type=int, default=6, help="Maximum candidate IDs kept for each S1 entity.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for generated TSV outputs.")
    parser.add_argument("--small-test", action="store_true", help="Run a small, reproducible smoke test instead of the full dataset pipeline.")
    args = parser.parse_args()

    if args.small_test:
        run_small_test()
        return

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_stage("loading training data")
    train_s1 = read_tsv(TRAIN_DIR / "train_source1.tsv")
    train_s2 = read_tsv(TRAIN_DIR / "train_source2.tsv")
    train_s3 = read_tsv(TRAIN_DIR / "train_source3.tsv")
    truth_df = read_tsv(TRAIN_DIR / "train_ground_truth.tsv")
    log_stage(f"training data ready: S1={len(train_s1)}, S2={len(train_s2)}, S3={len(train_s3)}")

    log_stage("building candidate index")
    pair_df, valid_df, _ = build_train_dataset(
        train_s1,
        train_s2,
        train_s3,
        truth_df,
        max_s1=None if args.train_max_s1 == 0 else args.train_max_s1,
        max_candidates=args.candidate_limit,
    )
    feature_cols = [c for c in pair_df.columns if c not in {"source1_entity_id", "candidate_entity_id", "label"}]
    log_stage(f"training candidates prepared: {len(pair_df)} pairs")

    log_stage("training model")
    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=200, random_state=42)
    model.fit(pair_df[feature_cols].to_numpy(dtype=float), pair_df["label"].to_numpy(dtype=int))
    threshold, val_score = select_threshold(model, valid_df, truth_df)
    log_stage(f"validation complete: threshold={threshold:.3f}, macro_F0.5={val_score:.4f}")

    log_stage("processing test S1 records")
    candidate_df, match_df = generate_test_outputs(model, threshold, max_candidates=args.candidate_limit)
    log_stage(f"writing outputs to {output_dir}")
    candidate_df.to_csv(output_dir / "candidate_pairs.tsv", sep="\t", index=False)
    match_df.to_csv(output_dir / "matching_results.tsv", sep="\t", index=False)
    log_stage(f"saved {len(candidate_df)} candidate rows and {len(match_df)} prediction rows")


if __name__ == "__main__":
    main()

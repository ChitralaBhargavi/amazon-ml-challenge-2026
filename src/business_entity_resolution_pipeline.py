"""Business Entity Resolution pipeline for the Amazon ML Challenge 2026.

This version keeps the real challenge data and uses a practical blocking +
feature-based model design that is fast enough to run on a single workstation.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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


@dataclass
class Record:
    entity_id: str
    business_name: str
    business_address: str
    country: str
    name_tokens: List[str]
    address_tokens: List[str]
    normalized_name: str
    normalized_address: str


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if path.name.endswith("ground_truth.tsv"):
        required = ["source1_entity_id", "matched_entity_ids"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing ground-truth columns {missing} in {path}")
        return df

    required = ["entity_id", "business_name", "business_address", "country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df


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


def tokenise(raw: Optional[str], *, drop_stopwords: bool = False) -> List[str]:
    cleaned = clean_text(raw)
    if not cleaned:
        return []
    tokens = [t for t in cleaned.split() if t]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [t for t in tokens if len(t) > 1]


def to_record(row: pd.Series) -> Record:
    name_clean = clean_text(row.get("business_name"), is_name=True)
    address_clean = clean_text(row.get("business_address"))
    name_tokens = tokenise(name_clean, drop_stopwords=True)
    address_tokens = tokenise(address_clean)
    return Record(
        entity_id=str(row["entity_id"]),
        business_name=str(row.get("business_name") or ""),
        business_address=str(row.get("business_address") or ""),
        country=str(row.get("country") or ""),
        name_tokens=name_tokens,
        address_tokens=address_tokens,
        normalized_name=name_clean,
        normalized_address=address_clean,
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


def generate_candidates(s1: Record, candidate_records: Sequence[Record], index: Dict[Tuple[str, str], Set[str]], max_candidates: int = 8) -> List[str]:
    if not s1.country:
        return []

    candidate_ids: Set[str] = set()
    for token in s1.name_tokens[:6] + s1.address_tokens[:6]:
        if not token:
            continue
        candidate_ids |= index.get((s1.country, token), set())
    if s1.normalized_name:
        candidate_ids |= index.get((s1.country, s1.normalized_name), set())
    if s1.name_tokens:
        bigram = " ".join(s1.name_tokens[:2])
        if bigram:
            candidate_ids |= index.get((s1.country, bigram), set())

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
    for _, row in truth_df.iterrows():
        sid = str(row["source1_entity_id"])
        ids = [x.strip() for x in str(row["matched_entity_ids"] or "").split(",") if x.strip()]
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

    candidate_records = [to_record(r) for _, r in pd.concat([train_s2, train_s3], ignore_index=True).iterrows()]
    index = build_var_index(candidate_records)
    truth_map = build_truth_map(truth_df)
    rec_by_id = {rec.entity_id: rec for rec in candidate_records}

    rows: List[dict] = []
    candidate_map: Dict[str, List[str]] = {}
    for _, row in train_s1.iterrows():
        s1 = to_record(row)
        cand_ids = generate_candidates(s1, candidate_records, index, max_candidates=max_candidates)
        candidate_map[s1.entity_id] = cand_ids
        for cand_id in cand_ids:
            cand = rec_by_id.get(cand_id)
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
    s1_ids = train_s1["entity_id"].tolist()
    _, valid_ids = train_test_split(s1_ids, test_size=0.15, random_state=42)

    valid_rows: List[dict] = []
    for sid in valid_ids:
        s1_row = train_s1.loc[train_s1["entity_id"] == sid]
        if s1_row.empty:
            continue
        s1 = to_record(s1_row.iloc[0])
        for cand_id in candidate_map.get(sid, []):
            cand = rec_by_id.get(cand_id)
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

    return pair_df, valid_df, candidate_map


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
            values.append(f05_entity(truth_map.get(sid, set()), pred_ids))
        metric = float(np.mean(values)) if values else 1.0
        if metric > best_score:
            best_score = metric
            best_threshold = float(threshold)
    return best_threshold, best_score


def generate_test_outputs(model: HistGradientBoostingClassifier, threshold: float, max_candidates: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_s1 = read_tsv(TEST_DIR / "test_source1.tsv")
    test_s2 = read_tsv(TEST_DIR / "test_source2.tsv")
    test_s3 = read_tsv(TEST_DIR / "test_source3.tsv")
    records = [to_record(r) for _, r in pd.concat([test_s2, test_s3], ignore_index=True).iterrows()]
    rec_by_id = {rec.entity_id: rec for rec in records}
    index = build_var_index(records)

    feature_cols = [
        "same_country", "same_name_exact", "same_address_exact", "name_jaccard", "address_jaccard",
        "name_overlap", "address_overlap", "name_char_sim", "address_char_sim",
        "first_token_match", "last_token_match", "name_len_ratio", "address_len_ratio",
        "num_name_tokens_1", "num_name_tokens_2", "num_address_tokens_1", "num_address_tokens_2",
    ]

    candidate_rows: List[dict] = []
    match_rows: List[dict] = []
    for _, row in test_s1.iterrows():
        s1 = to_record(row)
        cand_ids = generate_candidates(s1, records, index, max_candidates=max_candidates)
        candidate_rows.append({"source1_entity_id": s1.entity_id, "candidate_entity_ids": ",".join(cand_ids)})

        if not cand_ids:
            match_rows.append({"source1_entity_id": s1.entity_id, "matched_entity_ids": ""})
            continue

        hits = []
        for cand_id in cand_ids:
            cand = rec_by_id.get(cand_id)
            if cand is None:
                continue
            feat = pair_features(s1, cand)
            feat_arr = np.asarray([[float(feat.get(col, 0.0)) for col in feature_cols]], dtype=float)
            score = float(model.predict_proba(feat_arr)[0, 1])
            if score >= threshold:
                hits.append(cand_id)
        match_rows.append({"source1_entity_id": s1.entity_id, "matched_entity_ids": ",".join(sorted(set(hits)))})

    return pd.DataFrame(candidate_rows), pd.DataFrame(match_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("--train-max-s1", type=int, default=2000, help="Maximum S1 rows used for training; set 0 to use all rows.")
    parser.add_argument("--candidate-limit", type=int, default=6, help="Maximum candidate IDs kept for each S1 entity.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for generated TSV outputs.")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_s1 = read_tsv(TRAIN_DIR / "train_source1.tsv")
    train_s2 = read_tsv(TRAIN_DIR / "train_source2.tsv")
    train_s3 = read_tsv(TRAIN_DIR / "train_source3.tsv")
    truth_df = read_tsv(TRAIN_DIR / "train_ground_truth.tsv")

    print(f"Training subset size: {len(train_s1)} rows")
    pair_df, valid_df, _ = build_train_dataset(
        train_s1,
        train_s2,
        train_s3,
        truth_df,
        max_s1=None if args.train_max_s1 == 0 else args.train_max_s1,
        max_candidates=args.candidate_limit,
    )
    feature_cols = [c for c in pair_df.columns if c not in {"source1_entity_id", "candidate_entity_id", "label"}]

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=200, random_state=42)
    model.fit(pair_df[feature_cols].to_numpy(dtype=float), pair_df["label"].to_numpy(dtype=int))
    threshold, val_score = select_threshold(model, valid_df, truth_df)
    print(f"Validation threshold chosen: {threshold:.3f}; macro F0.5 ≈ {val_score:.4f}")

    candidate_df, match_df = generate_test_outputs(model, threshold, max_candidates=args.candidate_limit)
    candidate_df.to_csv(output_dir / "candidate_pairs.tsv", sep="\t", index=False)
    match_df.to_csv(output_dir / "matching_results.tsv", sep="\t", index=False)
    print(f"Saved {len(candidate_df)} candidate rows and {len(match_df)} prediction rows to {output_dir}")


if __name__ == "__main__":
    main()

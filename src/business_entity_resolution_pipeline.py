"""Business Entity Resolution pipeline for the Amazon ML Challenge 2026.

Scalable, precision-oriented entity resolution pipeline.

Key design choices:
- Candidate data are stored in SQLite instead of millions of Python Record objects.
- SQLite FTS5 provides country-aware token blocking without a giant Python dict/set index.
- Exact normalized-name/address indexes provide high-precision shortcuts.
- Test S1 is processed in streaming chunks, so the full test set is never held in RAM.
- Validation is split by S1 entity before model fitting to avoid leakage.
- Candidate recall is reported before the model is trusted.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unicodedata
import zlib
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
CACHE_DIR = OUTPUT_DIR / "entity_resolution_cache"

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", "pvt",
    "private", "plc", "co", "company", "companies", "group", "grp", "llp", "sarl",
    "sas", "sa", "service", "services", "solutions", "consultants", "consulting",
    "trading", "foundation", "enterprise", "enterprises", "industries",
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_SLASH_RE = re.compile(r"[/_\\|~]+")
_SPACE_RE = re.compile(r"\s+")
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(sorted((re.escape(x) for x in LEGAL_SUFFIXES), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

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


def read_tsv(
    path: Path,
    *,
    nrows: Optional[int] = None,
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
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


def iter_source_records(path: Path, *, chunk_size: int = 100_000) -> Iterator[Record]:
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


def iter_source_rows(path: Path, *, chunk_size: int = 100_000):
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
        yield chunk


def ascii_normalize(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    return text.encode("ascii", "ignore").decode("ascii")


def clean_text(raw: Optional[str], *, is_name: bool = False) -> str:
    text = ascii_normalize(raw or "")
    text = text.lower().strip()
    text = text.replace("&", " and ").replace("+", " ")
    text = _SLASH_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if is_name:
        text = _LEGAL_SUFFIX_RE.sub(" ", text)
        text = _SPACE_RE.sub(" ", text).strip()
    return text


def tokenise(raw: Optional[str], *, drop_stopwords: bool = False) -> Tuple[str, ...]:
    cleaned = clean_text(raw)
    if not cleaned:
        return ()
    tokens = [t for t in cleaned.split() if t]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tuple(t for t in tokens if len(t) > 1)


def to_record_values(
    entity_id: object,
    business_name: object,
    business_address: object,
    country: object,
) -> Record:
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
    """Fast character similarity.

    quick_ratio is deliberately used instead of the much slower full
    SequenceMatcher ratio because this function is evaluated millions of times.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, a, b, autojunk=False).quick_ratio()


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


def _fts_escape_token(token: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]", "", token)
    return token


def _fts_country(country: str) -> str:
    parts = [_fts_escape_token(x) for x in clean_text(country).split() if x]
    parts = [x for x in parts if x]
    if not parts:
        return ""
    if len(parts) == 1:
        return f"country:{parts[0]}"
    return "country:(" + " ".join(parts) + ")"


class CandidateStore:
    """Disk-backed candidate store.

    The store keeps normalized records in SQLite and a contentless FTS5
    blocking index. This replaces a multi-million-entry Python dict/set index.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-100000")
        self.conn.execute("PRAGMA locking_mode=EXCLUSIVE")

    @classmethod
    def build(
        cls,
        db_path: Path,
        source_paths: Sequence[Path],
        *,
        chunk_size: int = 50_000,
    ) -> "CandidateStore":
        if db_path.exists():
            db_path.unlink()

        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = cls(db_path)

        store.conn.executescript(
            """
            CREATE TABLE records (
                row_id INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL UNIQUE,
                country TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                normalized_address TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE fts USING fts5(
                country,
                name_tokens,
                address_tokens,
                content=''
            );
            """
        )

        row_id = 0
        total = 0

        for path in source_paths:
            log_stage(f"indexing {path.name}")
            for df in iter_source_rows(path, chunk_size=chunk_size):
                records_rows = []
                fts_rows = []

                for row in df.itertuples(index=False, name=None):
                    rec = to_record_values(row[0], row[1], row[2], row[3])
                    row_id += 1

                    # FTS uses stopword-free name tokens and address tokens
                    # without the most common road/address words.
                    fts_name = " ".join(rec.name_tokens)
                    fts_addr = " ".join(
                        t for t in rec.address_tokens if t not in STOPWORDS
                    )

                    records_rows.append(
                        (
                            row_id,
                            rec.entity_id,
                            rec.country,
                            rec.normalized_name,
                            rec.normalized_address,
                        )
                    )
                    fts_rows.append(
                        (row_id, rec.country, fts_name, fts_addr)
                    )

                with store.conn:
                    store.conn.executemany(
                        """
                        INSERT INTO records
                        (row_id, entity_id, country, normalized_name, normalized_address)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        records_rows,
                    )
                    store.conn.executemany(
                        """
                        INSERT INTO fts(rowid, country, name_tokens, address_tokens)
                        VALUES (?, ?, ?, ?)
                        """,
                        fts_rows,
                    )

                total += len(records_rows)
                if total % 500_000 < len(records_rows):
                    log_stage(f"candidate index rows built: {total:,}")

        log_stage("building exact-match indexes")
        with store.conn:
            store.conn.execute(
                "CREATE INDEX idx_records_country_name ON records(country, normalized_name)"
            )
            store.conn.execute(
                "CREATE INDEX idx_records_country_address ON records(country, normalized_address)"
            )
            store.conn.execute(
                "CREATE INDEX idx_records_entity ON records(entity_id)"
            )
        store.conn.execute("ANALYZE")
        log_stage(f"candidate index ready: {total:,} records")
        return store

    @classmethod
    def build_training_subset(
        cls,
        db_path: Path,
        source_paths: Sequence[Path],
        required_ids: Set[str],
        *,
        sample_mod: int = 40,
        chunk_size: int = 100_000,
    ) -> "CandidateStore":
        """Build a compact training candidate store.

        Keep every known positive entity from the sampled S1 ground truth plus
        a deterministic background sample. Stream records so the full 10M-row
        candidate corpus is never materialized in memory.
        """
        if db_path.exists():
            db_path.unlink()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = cls(db_path)
        store.conn.executescript("""
            CREATE TABLE records (
                row_id INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL UNIQUE,
                country TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                normalized_address TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE fts USING fts5(
                country, name_tokens, address_tokens, content=''
            );
        """)
        row_id = 0
        total = 0
        required_ids = {str(x) for x in required_ids}

        for path in source_paths:
            log_stage(f"training subset scan {path.name}")
            records_rows = []
            fts_rows = []

            for rec in iter_source_records(path, chunk_size=chunk_size):
                keep = rec.entity_id in required_ids
                if not keep and sample_mod > 0:
                    keep = (zlib.crc32(rec.entity_id.encode("utf-8", "ignore")) % sample_mod) == 0
                if not keep:
                    continue

                row_id += 1
                fts_name = " ".join(rec.name_tokens)
                fts_addr = " ".join(t for t in rec.address_tokens if t not in STOPWORDS)
                records_rows.append((
                    row_id, rec.entity_id, rec.country,
                    rec.normalized_name, rec.normalized_address
                ))
                fts_rows.append((row_id, rec.country, fts_name, fts_addr))

                if len(records_rows) >= chunk_size:
                    with store.conn:
                        store.conn.executemany(
                            "INSERT INTO records(row_id, entity_id, country, normalized_name, normalized_address) VALUES (?, ?, ?, ?, ?)",
                            records_rows,
                        )
                        store.conn.executemany(
                            "INSERT INTO fts(rowid, country, name_tokens, address_tokens) VALUES (?, ?, ?, ?)",
                            fts_rows,
                        )
                    total += len(records_rows)
                    records_rows = []
                    fts_rows = []

            if records_rows:
                with store.conn:
                    store.conn.executemany(
                        "INSERT INTO records(row_id, entity_id, country, normalized_name, normalized_address) VALUES (?, ?, ?, ?, ?)",
                        records_rows,
                    )
                    store.conn.executemany(
                        "INSERT INTO fts(rowid, country, name_tokens, address_tokens) VALUES (?, ?, ?, ?)",
                        fts_rows,
                    )
                total += len(records_rows)

        with store.conn:
            store.conn.execute("CREATE INDEX idx_records_country_name ON records(country, normalized_name)")
            store.conn.execute("CREATE INDEX idx_records_country_address ON records(country, normalized_address)")
            store.conn.execute("CREATE INDEX idx_records_entity ON records(entity_id)")
        store.conn.execute("ANALYZE")
        log_stage(f"training candidate subset ready: {total:,} records")
        return store

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def delete(self) -> None:
        self.close()
        try:
            self.db_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _fetch_records(self, row_ids: Sequence[int]) -> Dict[str, Record]:
        if not row_ids:
            return {}

        out: Dict[str, Record] = {}
        # SQLite has a parameter limit on some builds; batches keep this safe.
        for start in range(0, len(row_ids), 400):
            batch = row_ids[start:start + 400]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"""
                SELECT row_id, entity_id, country, normalized_name, normalized_address
                FROM records
                WHERE row_id IN ({placeholders})
                """,
                tuple(batch),
            ).fetchall()

            for _, entity_id, country, name, address in rows:
                out[str(entity_id)] = Record(
                    entity_id=str(entity_id),
                    country=str(country),
                    name_tokens=tokenise(name, drop_stopwords=True),
                    address_tokens=tokenise(address),
                    normalized_name=str(name),
                    normalized_address=str(address),
                )
        return out

    def _exact_ids(
        self,
        country: str,
        field: str,
        value: str,
        limit: int,
    ) -> List[int]:
        if not country or not value:
            return []
        if field == "name":
            sql = """
                SELECT row_id FROM records
                WHERE country=? AND normalized_name=?
                LIMIT ?
            """
        else:
            sql = """
                SELECT row_id FROM records
                WHERE country=? AND normalized_address=?
                LIMIT ?
            """
        return [int(x[0]) for x in self.conn.execute(sql, (country, value, limit)).fetchall()]

    def _prefix_ids(self, s1: Record, limit: int = 160) -> List[int]:
        """Fast B-tree prefix blocking using the existing country/name/address indexes.

        This intentionally avoids per-S1 FTS ranking/search. The records table already
        has country+normalized_name and country+normalized_address indexes, so bounded
        lexicographic prefix ranges are much cheaper than FTS MATCH over millions of rows.
        """
        if not s1.country:
            return []

        probes: List[Tuple[str, str]] = []

        # Name probes: beginning of the normalized full name plus first/last token.
        name_prefixes: List[str] = []
        if s1.normalized_name:
            name_prefixes.append(s1.normalized_name[:5])
        if s1.name_tokens:
            name_prefixes.append(s1.name_tokens[0][:5])
            if len(s1.name_tokens) > 1:
                name_prefixes.append(s1.name_tokens[-1][:5])

        # Address probe: beginning of normalized address and first useful token.
        addr_prefixes: List[str] = []
        if s1.normalized_address:
            addr_prefixes.append(s1.normalized_address[:6])
        for tok in s1.address_tokens:
            if tok not in STOPWORDS and len(tok) >= 3:
                addr_prefixes.append(tok[:5])
                break

        seen_prefixes: Set[Tuple[str, str]] = set()
        for p in name_prefixes:
            if len(p) >= 3 and ("name", p) not in seen_prefixes:
                seen_prefixes.add(("name", p))
                probes.append(("name", p))
        for p in addr_prefixes:
            if len(p) >= 3 and ("address", p) not in seen_prefixes:
                seen_prefixes.add(("address", p))
                probes.append(("address", p))

        if not probes:
            return []

        per_probe = max(20, min(60, limit // max(1, len(probes))))
        row_ids: List[int] = []
        for field, prefix in probes:
            upper = prefix + "\uffff"
            if field == "name":
                sql = """
                    SELECT row_id FROM records
                    WHERE country=? AND normalized_name>=? AND normalized_name<?
                    LIMIT ?
                """
            else:
                sql = """
                    SELECT row_id FROM records
                    WHERE country=? AND normalized_address>=? AND normalized_address<?
                    LIMIT ?
                """
            rows = self.conn.execute(
                sql, (s1.country, prefix, upper, per_probe)
            ).fetchall()
            row_ids.extend(int(x[0]) for x in rows)
            if len(row_ids) >= limit:
                break

        return row_ids[:limit]

    def candidates(
        self,
        s1: Record,
        *,
        max_candidates: int = 12,
        retrieval_limit: int = 160,
    ) -> List[Record]:
        if not s1.country and not s1.normalized_name and not s1.normalized_address:
            return []

        row_ids: List[int] = []

        # Exact blocks are deliberately kept separate and always retained.
        row_ids.extend(self._exact_ids(s1.country, "name", s1.normalized_name, 40))
        row_ids.extend(self._exact_ids(s1.country, "address", s1.normalized_address, 40))

        # Fast prefix blocking.  This replaces the per-S1 FTS MATCH query, which
        # became the dominant runtime cost on the 9.97M-row test index.
        if len(set(row_ids)) < max_candidates:
            row_ids.extend(self._prefix_ids(s1, retrieval_limit))

        # De-duplicate while preserving the high-value exact blocks first.
        seen: Set[int] = set()
        unique_ids: List[int] = []
        for rid in row_ids:
            if rid not in seen:
                seen.add(rid)
                unique_ids.append(rid)

        if not unique_ids:
            return []

        records = self._fetch_records(unique_ids)

        # First stage: cheap similarity. This avoids expensive character
        # similarity on the entire FTS result set.
        cheap: List[Tuple[float, Record]] = []
        for cand in records.values():
            if s1.country and cand.country != s1.country:
                continue

            name_exact = bool(
                s1.normalized_name
                and cand.normalized_name
                and s1.normalized_name == cand.normalized_name
            )
            addr_exact = bool(
                s1.normalized_address
                and cand.normalized_address
                and s1.normalized_address == cand.normalized_address
            )

            name_j = jaccard(s1.name_tokens, cand.name_tokens)
            name_o = overlap_score(s1.name_tokens, cand.name_tokens)
            addr_j = jaccard(s1.address_tokens, cand.address_tokens)
            addr_o = overlap_score(s1.address_tokens, cand.address_tokens)

            cheap_score = (
                8.0 * float(name_exact)
                + 5.0 * float(addr_exact)
                + 3.0 * name_j
                + 2.0 * name_o
                + 1.5 * addr_j
                + 1.0 * addr_o
            )
            cheap.append((cheap_score, cand))

        cheap.sort(key=lambda x: x[0], reverse=True)

        # Only a small high-quality set reaches character similarity/model.
        top = cheap[: max(30, max_candidates * 3)]

        scored: List[Tuple[float, Record]] = []
        for cheap_score, cand in top:
            name_char = seq_sim(s1.normalized_name, cand.normalized_name)
            addr_char = seq_sim(s1.normalized_address, cand.normalized_address)

            final_rank = (
                cheap_score
                + 1.5 * name_char
                + 0.5 * addr_char
            )
            scored.append((final_rank, cand))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep a little extra room when several exact-name records exist.
        return [cand for _, cand in scored[:max_candidates]]


def build_var_index(records: Sequence[Record]) -> Dict[Tuple[str, str], Set[str]]:
    """Legacy small-data index retained for the smoke test."""
    index: Dict[Tuple[str, str], Set[str]] = {}
    for rec in records:
        if not rec.country:
            continue
        for token in rec.name_tokens + rec.address_tokens[:6]:
            if token:
                index.setdefault((rec.country, token), set()).add(rec.entity_id)
        if rec.normalized_name:
            index.setdefault((rec.country, rec.normalized_name), set()).add(rec.entity_id)
        if rec.name_tokens:
            bigram = " ".join(rec.name_tokens[:2])
            if bigram:
                index.setdefault((rec.country, bigram), set()).add(rec.entity_id)
    return index


def generate_candidates(
    s1: Record,
    candidate_records,
    index,
    max_candidates: int = 12,
) -> List[str]:
    """Compatibility wrapper for the original in-memory API.

    Full-scale execution uses CandidateStore directly.
    """
    if isinstance(candidate_records, CandidateStore):
        return [r.entity_id for r in candidate_records.candidates(
            s1, max_candidates=max_candidates
        )]

    if not s1.country:
        return []

    rec_by_id = (
        candidate_records
        if isinstance(candidate_records, dict)
        else {rec.entity_id: rec for rec in candidate_records}
    )

    candidate_ids: Set[str] = set()
    for token in s1.name_tokens[:6] + s1.address_tokens[:6]:
        if token:
            candidate_ids.update(index.get((s1.country, token), set()))

    if s1.normalized_name:
        candidate_ids.update(index.get((s1.country, s1.normalized_name), set()))

    if s1.name_tokens:
        bigram = " ".join(s1.name_tokens[:2])
        if bigram:
            candidate_ids.update(index.get((s1.country, bigram), set()))

    scored: List[Tuple[float, str]] = []
    for cand_id in candidate_ids:
        cand = rec_by_id.get(cand_id)
        if cand is None or cand.country != s1.country:
            continue

        name_j = jaccard(s1.name_tokens, cand.name_tokens)
        addr_o = overlap_score(s1.address_tokens, cand.address_tokens)
        score = (
            5.0 * float(s1.normalized_name == cand.normalized_name and s1.normalized_name)
            + 3.0 * name_j
            + 1.5 * addr_o
            + 1.0 * seq_sim(s1.normalized_name, cand.normalized_name)
        )
        scored.append((score, cand_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:max_candidates]]


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


def _candidate_recall(
    candidate_ids_by_s1: Dict[str, List[str]],
    truth_map: Dict[str, Set[str]],
) -> float:
    values = []
    for sid, true_ids in truth_map.items():
        if not true_ids:
            continue
        candidates = set(candidate_ids_by_s1.get(sid, []))
        values.append(len(true_ids & candidates) / len(true_ids))
    return float(np.mean(values)) if values else 0.0


def build_train_dataset_from_store(
    train_s1: pd.DataFrame,
    truth_df: pd.DataFrame,
    store: CandidateStore,
    *,
    max_candidates: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    truth_map = build_truth_map(truth_df)

    rows: List[dict] = []
    candidate_ids_by_s1: Dict[str, List[str]] = {}

    for row in train_s1.itertuples(index=False, name=None):
        s1 = to_record_values(row[0], row[1], row[2], row[3])
        candidates = store.candidates(s1, max_candidates=max_candidates)
        cand_ids = [c.entity_id for c in candidates]
        candidate_ids_by_s1[s1.entity_id] = cand_ids

        for cand in candidates:
            feat = pair_features(s1, cand)
            feat["source1_entity_id"] = s1.entity_id
            feat["candidate_entity_id"] = cand.entity_id
            feat["label"] = 1 if cand.entity_id in truth_map.get(s1.entity_id, set()) else 0
            rows.append(feat)

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise RuntimeError("No training pairs were generated from the candidate store.")

    feature_cols = [
        c for c in pair_df.columns
        if c not in {"source1_entity_id", "candidate_entity_id", "label"}
    ]

    s1_ids = train_s1["entity_id"].astype(str).tolist()
    train_ids, valid_ids = train_test_split(
        s1_ids,
        test_size=0.20,
        random_state=42,
    )

    train_df = pair_df[pair_df["source1_entity_id"].astype(str).isin(set(train_ids))].copy()
    valid_df = pair_df[pair_df["source1_entity_id"].astype(str).isin(set(valid_ids))].copy()

    if valid_df.empty:
        valid_df = pd.DataFrame(
            columns=[*feature_cols, "source1_entity_id", "candidate_entity_id", "label"]
        )

    recall = _candidate_recall(candidate_ids_by_s1, truth_map)
    log_stage(f"candidate recall on sampled training S1: {recall:.4f}")

    # Keep the full pair_df return for compatibility, but callers should fit
    # only on train_df to avoid validation leakage.
    train_df.attrs["fit_df"] = train_df
    return pair_df, valid_df, candidate_ids_by_s1


def select_threshold(
    model: HistGradientBoostingClassifier,
    valid_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> Tuple[float, float]:
    if valid_df.empty:
        return 0.5, 0.0

    feature_cols = [
        c for c in valid_df.columns
        if c not in {"source1_entity_id", "candidate_entity_id", "label"}
    ]
    scores = model.predict_proba(valid_df[feature_cols].to_numpy(dtype=float))[:, 1]

    scored_df = valid_df[["source1_entity_id", "candidate_entity_id"]].copy()
    scored_df["score"] = scores

    truth_map = build_truth_map(truth_df)
    best_threshold = 0.5
    best_score = -1.0

    # Fine-grained threshold sweep. F0.5 rewards precision, so a coarse
    # 0.05 grid can leave meaningful leaderboard points on the table.
    for threshold in np.linspace(0.05, 0.95, 91):
        values = []
        for sid, group in scored_df.groupby("source1_entity_id", sort=False):
            pred_ids = set(
                group.loc[group["score"] >= threshold, "candidate_entity_id"].astype(str)
            )
            values.append(f05_entity(truth_map.get(str(sid), set()), pred_ids))

        metric = float(np.mean(values)) if values else 0.0
        if metric > best_score:
            best_score = metric
            best_threshold = float(threshold)

    return best_threshold, best_score


FEATURE_COLS = [
    "same_country",
    "same_name_exact",
    "same_address_exact",
    "name_jaccard",
    "address_jaccard",
    "name_overlap",
    "address_overlap",
    "name_char_sim",
    "address_char_sim",
    "first_token_match",
    "last_token_match",
    "name_len_ratio",
    "address_len_ratio",
    "num_name_tokens_1",
    "num_name_tokens_2",
    "num_address_tokens_1",
    "num_address_tokens_2",
]


def generate_test_outputs_streaming(
    model: HistGradientBoostingClassifier,
    threshold: float,
    store: CandidateStore,
    test_s1_path: Path,
    matching_path: Path,
    candidate_path: Path,
    *,
    max_candidates: int = 12,
    chunk_size: int = 50_000,
) -> int:
    matching_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0

    with (
        matching_path.open("w", encoding="utf-8", newline="") as match_file,
        candidate_path.open("w", encoding="utf-8", newline="") as cand_file,
    ):
        match_writer = csv.writer(match_file, delimiter="\t", lineterminator="\n")
        cand_writer = csv.writer(cand_file, delimiter="\t", lineterminator="\n")

        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for df in iter_source_rows(test_s1_path, chunk_size=chunk_size):
            for row in df.itertuples(index=False, name=None):
                s1 = to_record_values(row[0], row[1], row[2], row[3])
                candidates = store.candidates(
                    s1,
                    max_candidates=max_candidates,
                )

                cand_ids = [c.entity_id for c in candidates]
                cand_writer.writerow([s1.entity_id, ",".join(cand_ids)])

                if not candidates:
                    match_writer.writerow([s1.entity_id, ""])
                    continue

                feature_matrix = np.asarray(
                    [
                        [
                            float(pair_features(s1, cand).get(col, 0.0))
                            for col in FEATURE_COLS
                        ]
                        for cand in candidates
                    ],
                    dtype=float,
                )

                scores = model.predict_proba(feature_matrix)[:, 1]
                hits = [
                    candidates[i].entity_id
                    for i, score in enumerate(scores)
                    if float(score) >= threshold
                ]

                match_writer.writerow(
                    [s1.entity_id, ",".join(sorted(set(hits)))]
                )
                rows_written += 1

            if rows_written and rows_written % 100_000 < len(df):
                log_stage(f"test S1 processed: {rows_written:,}")

    return rows_written


def run_small_test() -> None:
    sample_s1 = read_tsv(
        TRAIN_DIR / "train_source1.tsv",
        sample_n=25,
        random_state=42,
    )
    sample_s2 = read_tsv(
        TRAIN_DIR / "train_source2.tsv",
        sample_n=40,
        random_state=42,
    )
    sample_s3 = read_tsv(
        TRAIN_DIR / "train_source3.tsv",
        sample_n=40,
        random_state=42,
    )
    truth_df = read_tsv(TRAIN_DIR / "train_ground_truth.tsv")
    truth_df = truth_df[
        truth_df["source1_entity_id"].astype(str).isin(
            sample_s1["entity_id"].astype(str).tolist()
        )
    ].copy().reset_index(drop=True)

    print("=== Small-scale validation run ===")
    print(
        f"TSV loading: S1={len(sample_s1)}, S2={len(sample_s2)}, "
        f"S3={len(sample_s3)}, ground truth={len(truth_df)}"
    )

    small_dir = OUTPUT_DIR / "small_test"
    small_dir.mkdir(parents=True, exist_ok=True)
    db_path = small_dir / "candidate_store.sqlite"

    # Materialize only the tiny sampled candidate files for the smoke test.
    sample_s2_path = small_dir / "sample_train_source2.tsv"
    sample_s3_path = small_dir / "sample_train_source3.tsv"
    sample_s2.to_csv(sample_s2_path, sep="\t", index=False)
    sample_s3.to_csv(sample_s3_path, sep="\t", index=False)

    store = CandidateStore.build(
        db_path,
        [sample_s2_path, sample_s3_path],
        chunk_size=1_000,
    )

    try:
        pair_df, valid_df, candidate_map = build_train_dataset_from_store(
            sample_s1,
            truth_df,
            store,
            max_candidates=12,
        )

        feature_cols = [
            c for c in pair_df.columns
            if c not in {"source1_entity_id", "candidate_entity_id", "label"}
        ]
        train_df = pair_df.attrs.get("fit_df", pair_df)

        print(
            f"Normalization/feature extraction: {len(feature_cols)} "
            f"feature columns, {len(pair_df)} candidate pairs"
        )

        model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            max_iter=80,
            random_state=42,
        )
        model.fit(
            train_df[feature_cols].to_numpy(dtype=float),
            train_df["label"].to_numpy(dtype=int),
        )

        threshold, macro_f05 = select_threshold(model, valid_df, truth_df)
        print(
            f"Model training + prediction + F0.5: "
            f"threshold={threshold:.3f}, macro_F0.5={macro_f05:.4f}"
        )

        test_s1 = read_tsv(
            TEST_DIR / "test_source1.tsv",
            sample_n=12,
            random_state=7,
        )
        test_s2 = read_tsv(
            TEST_DIR / "test_source2.tsv",
            sample_n=18,
            random_state=7,
        )
        test_s3 = read_tsv(
            TEST_DIR / "test_source3.tsv",
            sample_n=18,
            random_state=7,
        )

        # Build a tiny temporary store for the smoke test, so the same
        # disk-backed path used at scale is exercised.
        tiny_path = small_dir / "tiny_test_store.sqlite"
        sample_test_s2_path = small_dir / "sample_test_source2.tsv"
        sample_test_s3_path = small_dir / "sample_test_source3.tsv"
        test_s2.to_csv(sample_test_s2_path, sep="\t", index=False)
        test_s3.to_csv(sample_test_s3_path, sep="\t", index=False)

        tiny = CandidateStore.build(
            tiny_path,
            [sample_test_s2_path, sample_test_s3_path],
            chunk_size=1_000,
        )
        try:
            candidate_path = small_dir / "candidate_pairs.tsv"
            match_path = small_dir / "matching_results.tsv"

            # Write only the selected 12-row test sample.
            rows_c = []
            rows_m = []
            for row in test_s1.itertuples(index=False, name=None):
                s1 = to_record_values(row[0], row[1], row[2], row[3])
                candidates = tiny.candidates(s1, max_candidates=12)
                ids = [c.entity_id for c in candidates]
                rows_c.append(
                    {
                        "source1_entity_id": s1.entity_id,
                        "candidate_entity_ids": ",".join(ids),
                    }
                )
                if candidates:
                    X = np.asarray(
                        [
                            [
                                float(pair_features(s1, c).get(col, 0.0))
                                for col in feature_cols
                            ]
                            for c in candidates
                        ],
                        dtype=float,
                    )
                    scores = model.predict_proba(X)[:, 1]
                    hits = [
                        candidates[i].entity_id
                        for i, score in enumerate(scores)
                        if float(score) >= threshold
                    ]
                else:
                    hits = []
                rows_m.append(
                    {
                        "source1_entity_id": s1.entity_id,
                        "matched_entity_ids": ",".join(sorted(set(hits))),
                    }
                )

            pd.DataFrame(rows_c).to_csv(candidate_path, sep="\t", index=False)
            pd.DataFrame(rows_m).to_csv(match_path, sep="\t", index=False)

            # Preserve test files for the validator.
            test_dir = small_dir
            test_s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
            test_s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
            test_s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

            print(
                f"Output generation: {len(rows_c)} candidate rows, "
                f"{len(rows_m)} match rows"
            )

            validator = [
                sys.executable,
                str(ROOT / "utils" / "validate_submission.py"),
                "--matching",
                str(match_path),
                "--candidate",
                str(candidate_path),
                "--test-dir",
                str(test_dir),
            ]
            result = subprocess.run(
                validator,
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            print(result.stdout.strip())
            if result.stderr.strip():
                print(result.stderr.strip())
            print(f"Submission validation exit code: {result.returncode}")
            if result.returncode != 0:
                raise SystemExit(result.returncode)
        finally:
            tiny.delete()
    finally:
        store.delete()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scalable Business Entity Resolution pipeline"
    )
    parser.add_argument(
        "--train-max-s1",
        type=int,
        default=5000,
        help="Maximum S1 rows used for model training; set 0 to use all rows.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=12,
        help="Maximum candidate IDs retained per S1 entity.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for generated TSV outputs.",
    )
    parser.add_argument(
        "--small-test",
        action="store_true",
        help="Run a small reproducible smoke test instead of the full pipeline.",
    )
    args = parser.parse_args()

    if args.small_test:
        run_small_test()
        return

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- TRAIN ----------
    log_stage("loading training S1")
    if args.train_max_s1 == 0:
        train_s1 = read_tsv(TRAIN_DIR / "train_source1.tsv")
    else:
        # Read only the requested number of S1 rows. This avoids loading all
        # 2.2M S1 rows just to train a model.
        train_s1 = read_tsv(
            TRAIN_DIR / "train_source1.tsv",
            nrows=args.train_max_s1,
        )

    log_stage(f"training S1 ready: {len(train_s1):,}")
    log_stage("loading ground truth")
    truth_df = read_tsv(TRAIN_DIR / "train_ground_truth.tsv")
    truth_df = truth_df[
        truth_df["source1_entity_id"].astype(str).isin(
            train_s1["entity_id"].astype(str).tolist()
        )
    ].copy()
    log_stage(f"training ground truth ready: {len(truth_df):,}")

    train_db = CACHE_DIR / "train_candidates.sqlite"
    truth_ids = set()
    for _sid, _ids in build_truth_map(truth_df).items():
        truth_ids.update(_ids)
    log_stage("building compact training candidate subset")
    train_store = CandidateStore.build_training_subset(
        train_db,
        [
            TRAIN_DIR / "train_source2.tsv",
            TRAIN_DIR / "train_source3.tsv",
        ],
        truth_ids,
        sample_mod=40,
        chunk_size=100_000,
    )

    try:
        pair_df, valid_df, _ = build_train_dataset_from_store(
            train_s1,
            truth_df,
            train_store,
            max_candidates=args.candidate_limit,
        )

        feature_cols = [
            c for c in pair_df.columns
            if c not in {"source1_entity_id", "candidate_entity_id", "label"}
        ]
        fit_df = pair_df.attrs.get("fit_df", pair_df)

        log_stage(
            f"training candidates prepared: {len(pair_df):,} pairs "
            f"({len(fit_df):,} used for fitting)"
        )

        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=200,
            random_state=42,
        )
        model.fit(
            fit_df[feature_cols].to_numpy(dtype=float),
            fit_df["label"].to_numpy(dtype=int),
        )

        threshold, val_score = select_threshold(
            model,
            valid_df,
            truth_df,
        )
        log_stage(
            f"validation complete: threshold={threshold:.3f}, "
            f"macro_F0.5={val_score:.4f}"
        )
    finally:
        # The training database is no longer needed once the model is trained.
        train_store.delete()

    # ---------- TEST ----------
    test_db = CACHE_DIR / "test_candidates.sqlite"
    if test_db.exists():
        log_stage("reusing existing disk-backed test candidate index")
        test_store = CandidateStore(test_db)
    else:
        log_stage("building disk-backed test candidate index")
        test_store = CandidateStore.build(
            test_db,
            [
                TEST_DIR / "test_source2.tsv",
                TEST_DIR / "test_source3.tsv",
            ],
            chunk_size=50_000,
        )

    try:
        matching_path = output_dir / "matching_results.tsv"
        candidate_path = output_dir / "candidate_pairs.tsv"

        log_stage("processing test S1 records in streaming mode")
        count = generate_test_outputs_streaming(
            model,
            threshold,
            test_store,
            TEST_DIR / "test_source1.tsv",
            matching_path,
            candidate_path,
            max_candidates=args.candidate_limit,
            chunk_size=50_000,
        )

        log_stage(
            f"saved {count:,} processed S1 rows to {output_dir}"
        )

        # Run the official validator if it is present.
        validator = [
            sys.executable,
            str(ROOT / "utils" / "validate_submission.py"),
            "--matching",
            str(matching_path),
            "--candidate",
            str(candidate_path),
            "--test-dir",
            str(TEST_DIR),
        ]
        result = subprocess.run(
            validator,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    finally:
        test_store.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    import polars as pl_t


@dataclass(frozen=True)
class Splits:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def _candidate_base_urls(base_url: str) -> list[str]:
    base_url = base_url.rstrip("/")
    candidates = [base_url]
    prefix = "hf://datasets/"
    if base_url.startswith(prefix):
        # Convert:
        # hf://datasets/<owner>/<dataset>/path
        # -> https://huggingface.co/datasets/<owner>/<dataset>/resolve/main/path
        tail = base_url[len(prefix) :]
        parts = tail.split("/")
        if len(parts) >= 2:
            repo = "/".join(parts[:2])
            rest = "/".join(parts[2:])
            converted = f"https://huggingface.co/datasets/{repo}/resolve/main"
            if rest:
                converted = f"{converted}/{rest}"
            candidates.append(converted)

    # dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _load_synthea_from_base(base_url: str) -> tuple["pl_t.DataFrame", "pl_t.DataFrame"]:
    patients_lf = pl.scan_parquet(f"{base_url}/patients.parquet").select(
        [
            pl.col("Id").alias("patient_id"),
            pl.col("DEATHDATE"),
        ]
    )
    labels_df = (
        patients_lf.with_columns(
            pl.when(pl.col("DEATHDATE").is_not_null()).then(1).otherwise(0).alias("label")
        )
        .select(["patient_id", "label"])
        .collect()
    )

    events_lf = pl.scan_parquet(f"{base_url}/conditions.parquet").select(
        [
            pl.col("PATIENT").alias("patient_id"),
            pl.col("START").str.to_datetime().alias("timestamp"),
            pl.col("CODE").alias("event_code"),
        ]
    )
    events_df = (
        events_lf.sort(["patient_id", "timestamp"])
        .with_columns(
            pl.col("timestamp")
            .diff()
            .dt.total_days()
            .over("patient_id")
            .fill_null(0)
            .alias("time_delta")
        )
        .collect()
    )
    return labels_df, events_df


def load_synthea_parquet(
    base_url: str = "https://huggingface.co/datasets/richardyoung/synthea-575k-patients/resolve/main/data",
) -> tuple["pl_t.DataFrame", "pl_t.DataFrame"]:
    """
    Returns:
      labels_df: columns [patient_id, label]
      events_df: columns [patient_id, timestamp, event_code, time_delta]
    """
    if pl is None:
        raise ImportError("polars is required for load_synthea_parquet(). Install: pip install polars")
    errors: list[str] = []
    for candidate in _candidate_base_urls(base_url):
        try:
            return _load_synthea_from_base(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc!r}")
    raise RuntimeError(
        "Failed to load Synthea parquet from all candidate URLs:\n" + "\n".join(errors)
    )


def make_splits(
    labels_df,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Splits:
    # Implemented without sklearn to keep the project easy to run.
    all_ids = labels_df["patient_id"].to_list() if hasattr(labels_df, "__getitem__") else list(labels_df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_ids))
    n_train = int(round(len(all_ids) * train_frac))
    n_val = int(round(len(all_ids) * val_frac))
    train_ids = [all_ids[i] for i in perm[:n_train]]
    val_ids = [all_ids[i] for i in perm[n_train : n_train + n_val]]
    test_ids = [all_ids[i] for i in perm[n_train + n_val :]]
    return Splits(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def create_vocab(events_df) -> dict[str, int]:
    unique_codes = events_df["event_code"].unique().to_list()
    vocab = {code: i + 2 for i, code in enumerate(unique_codes)}
    vocab["[PAD]"] = 0
    vocab["[UNK]"] = 1
    return vocab


class MEDSDataset(Dataset):
    def __init__(
        self,
        events_df,
        labels_df,
        vocab: dict[str, int],
        target_ids: Iterable[str],
        max_len: int = 256,
        desc: str = "dataset",
    ):
        if pl is None:
            raise ImportError("polars is required for MEDSDataset. Install: pip install polars")
        self.max_len = int(max_len)

        target_ids = list(target_ids)
        subset_labels = labels_df.filter(pl.col("patient_id").is_in(target_ids))
        self.label_map: dict[str, int] = dict(zip(subset_labels["patient_id"], subset_labels["label"]))

        subset_events = events_df.filter(pl.col("patient_id").is_in(target_ids))
        grouped = subset_events.group_by("patient_id", maintain_order=True).agg(
            [pl.col("event_code"), pl.col("time_delta")]
        )

        self.patient_ids: list[str] = grouped["patient_id"].to_list()
        raw_codes: list[list[str]] = grouped["event_code"].to_list()
        self.times: list[list[float]] = grouped["time_delta"].to_list()

        self.codes: list[list[int]] = [
            [vocab.get(c, 1) for c in row] for row in tqdm(raw_codes, desc=f"Tokenize {desc}")
        ]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p_id = self.patient_ids[idx]
        codes = list(self.codes[idx])
        times = list(self.times[idx])

        if len(codes) > self.max_len:
            codes, times = codes[-self.max_len :], times[-self.max_len :]
            mask = [1] * self.max_len
        else:
            pad_len = self.max_len - len(codes)
            mask = [1] * len(codes) + [0] * pad_len
            codes += [0] * pad_len
            times += [0.0] * pad_len

        return {
            "x": torch.tensor(codes, dtype=torch.long),
            "times": torch.tensor(times, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(float(self.label_map[p_id]), dtype=torch.float32),
        }


class TinySequenceDataset(Dataset):
    """
    Pure-Python/NumPy synthetic dataset (no polars dependency).
    Tokens are already integers in [0, vocab_size).
    """

    def __init__(
        self,
        sequences: list[list[int]],
        time_deltas: list[list[float]],
        labels: list[int],
        *,
        max_len: int,
        pad_idx: int = 0,
    ):
        if not (len(sequences) == len(time_deltas) == len(labels)):
            raise ValueError("sequences, time_deltas, labels must have same length")
        self.sequences = sequences
        self.time_deltas = time_deltas
        self.labels = labels
        self.max_len = int(max_len)
        self.pad_idx = int(pad_idx)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        codes = list(self.sequences[idx])
        times = list(self.time_deltas[idx])
        if len(codes) > self.max_len:
            codes, times = codes[-self.max_len :], times[-self.max_len :]
            mask = [1] * self.max_len
        else:
            pad_len = self.max_len - len(codes)
            mask = [1] * len(codes) + [0] * pad_len
            codes += [self.pad_idx] * pad_len
            times += [0.0] * pad_len
        return {
            "x": torch.tensor(codes, dtype=torch.long),
            "times": torch.tensor(times, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        }


def make_tiny_sequences(
    n_patients: int = 512,
    max_events: int = 64,
    vocab_size: int = 200,
    seed: int = 42,
    *,
    pad_idx: int = 0,
    unk_idx: int = 1,
) -> tuple[list[list[int]], list[list[float]], list[int], Splits]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_patients).tolist()
    ids = [f"p{i:05d}" for i in range(n_patients)]
    rng2 = np.random.default_rng(seed + 1)
    perm = rng2.permutation(n_patients)
    n_train = int(round(n_patients * 0.8))
    n_val = int(round(n_patients * 0.1))
    splits = Splits(
        train_ids=[ids[i] for i in perm[:n_train]],
        val_ids=[ids[i] for i in perm[n_train : n_train + n_val]],
        test_ids=[ids[i] for i in perm[n_train + n_val :]],
    )

    seqs: list[list[int]] = []
    times: list[list[float]] = []
    for _ in range(n_patients):
        n = int(rng.integers(5, max_events))
        codes = rng.integers(2, vocab_size, size=n).tolist()
        gaps = rng.integers(0, 30, size=n).astype(float).tolist()
        seqs.append([pad_idx if c < 0 else int(c) for c in codes])
        times.append(gaps)
    return seqs, times, labels, splits


def make_tiny_synthetic(
    n_patients: int = 256,
    max_events: int = 64,
    vocab_size: int = 200,
    seed: int = 42,
) -> tuple["pl_t.DataFrame", "pl_t.DataFrame"]:
    """
    Small offline dataset useful for smoke tests without HF downloads.
    """
    if pl is None:
        raise ImportError("polars is required for make_tiny_synthetic(). Install: pip install polars")
    rng = np.random.default_rng(seed)
    patient_ids = [f"p{i:05d}" for i in range(n_patients)]
    labels = rng.integers(0, 2, size=n_patients).tolist()
    labels_df = pl.DataFrame({"patient_id": patient_ids, "label": labels})

    rows = []
    for pid, y in zip(patient_ids, labels):
        n = int(rng.integers(5, max_events))
        codes = rng.integers(2, vocab_size, size=n).tolist()
        gaps = rng.integers(0, 30, size=n).tolist()
        ts = np.cumsum(gaps).tolist()
        for c, t, g in zip(codes, ts, gaps):
            rows.append((pid, int(t), str(c), float(g), int(y)))

    events_df = pl.DataFrame(
        rows, schema=["patient_id", "timestamp_int", "event_code", "time_delta", "label_hint"]
    ).select(["patient_id", "event_code", "time_delta"])
    return labels_df, events_df


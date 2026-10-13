from pathlib import Path

import numpy as np
import pandas as pd


# Failure intervals reported by the MetroPT-3 dataset maintainers.  Both
# endpoints are excluded from analysis as part of the failure period.
FAILURE_PERIODS = (
    (pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-18 23:59:00")),
    (pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 06:00:00")),
    (pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 14:30:00")),
    (pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),
)


def outside_failure_periods(timestamps):
    """Return True for observations that are outside every failure period."""
    keep = np.ones(len(timestamps), dtype=bool)
    for start, end in FAILURE_PERIODS:
        keep &= ~((timestamps >= start) & (timestamps <= end))
    return keep


def _count_usable_rows(csv_path, chunk_size):
    usable_rows = 0
    for frame in pd.read_csv(
            csv_path, usecols=["timestamp"], chunksize=chunk_size):
        timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
        usable_rows += int(np.count_nonzero(outside_failure_periods(timestamps)))
    return usable_rows


def prepare_failure_free_dataset(
        csv_path, cache_path, feature_columns, chunk_size=50_000):
    """Open a disk-backed feature matrix with all known failures removed.

    The CSV is processed in chunks.  The resulting NumPy cache contains only
    feature columns and can be sliced without loading the whole dataset into
    memory.
    """
    csv_path = Path(csv_path)
    cache_path = Path(cache_path)

    if (not cache_path.exists()
            or cache_path.stat().st_mtime < csv_path.stat().st_mtime):
        usable_rows = _count_usable_rows(csv_path, chunk_size)
        print(
            f"Building failure-free disk cache {cache_path} "
            f"({usable_rows:,} retained rows)..."
        )
        cached = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float64,
            shape=(usable_rows, len(feature_columns)),
        )

        offset = 0
        columns = ["timestamp", *feature_columns]
        for frame in pd.read_csv(
                csv_path,
                usecols=columns,
                chunksize=chunk_size):
            timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
            keep = outside_failure_periods(timestamps)
            values = frame.loc[keep, feature_columns].to_numpy(
                dtype=np.float64, copy=False
            )
            end = offset + len(values)
            cached[offset:end] = values
            offset = end

        if offset != usable_rows:
            raise RuntimeError(
                f"Expected to retain {usable_rows} rows, but wrote {offset}"
            )
        cached.flush()
        del cached

    return np.load(cache_path, mmap_mode="r")


def read_window(dataset, start, size):
    """Return a disk-backed, contiguous feature window."""
    if start < 0 or size < 0:
        raise ValueError("start and size must be non-negative")
    window = dataset[start:start + size]
    if len(window) != size:
        raise ValueError(f"Requested {size} rows at {start}, got {len(window)}")
    return window

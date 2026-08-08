"""Distributed feature pipeline (Module M1 at scale) — PySpark ETL.

Extracts the hand-crafted feature vector for every audio file in a directory
using Spark, reduces each to a `pca_dim` vector, and writes a Parquet feature
store. This mirrors the FMA 100k-track pipeline; on a laptop it happily runs on
the demo clips with a local[*] Spark session.

Runs only when pyspark is installed. Windows note: Spark needs HADOOP_HOME +
winutils.exe configured; the code checks and prints guidance rather than crashing.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

from ..config import Config
from ..features.extract import extract_feature_sequence, load_audio
from ..platform_utils import Platform


def _windows_spark_ready() -> bool:
    return bool(os.environ.get("HADOOP_HOME"))


def _detect_java_major() -> int | None:
    """Return the major Java version (e.g. 11, 17) or None if Java isn't found."""
    import re
    import subprocess
    try:
        out = subprocess.run(
            ["java", "-version"], capture_output=True, text=True
        ).stderr
    except FileNotFoundError:
        return None
    # Handles both "1.8.0_xxx" (=> 8) and "17.0.1" style version strings.
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out)
    if not m:
        return None
    major = int(m.group(1))
    if major == 1 and m.group(2):   # legacy "1.8" -> 8
        major = int(m.group(2))
    return major


def _preflight_java() -> None:
    """Fail early with an actionable message instead of JAVA_GATEWAY_EXITED."""
    major = _detect_java_major()
    if major is None:
        raise RuntimeError(
            "Java was not found on PATH, but PySpark needs a JDK. On macOS:\n"
            "  brew install openjdk@17\n"
            '  export JAVA_HOME="$(/usr/libexec/java_home -v 17)"\n'
            "Then re-run. (The CNN/LSTM/UI/API paths do NOT need Java — only "
            "the `features` Spark step does.)"
        )
    # PySpark 3.5 supports Java 8/11/17; 4.x needs 17+. PySpark is pinned <4.0,
    # so 11 is fine, but Java 21+/25 can still trip older Spark — warn on the
    # high end.
    if major < 8:
        raise RuntimeError(
            f"Java {major} is too old for PySpark. Install Java 17 and set "
            f'JAVA_HOME (e.g. `export JAVA_HOME="$(/usr/libexec/java_home -v 17)"`).'
        )
    if major > 17:
        print(
            f"[spark] WARNING: detected Java {major}. PySpark works best on Java "
            f"11 or 17. If you hit a gateway error, run: "
            f'export JAVA_HOME="$(/usr/libexec/java_home -v 17)"'
        )


def _feature_row(path: str, cfg: Config):
    """Compute a single aggregated feature vector for one audio file."""
    try:
        y = load_audio(path, cfg.audio.sample_rate)
        seq = extract_feature_sequence(y, cfg)     # (W, D)
        vec = seq.mean(axis=0)                     # aggregate over windows
        # Pad/truncate to pca_dim. A production pipeline would fit PCA; this
        # stays deterministic so no fitted transformer is needed on disk.
        dim = int(cfg.features.pca_dim)
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)))
        else:
            vec = vec[:dim]
        return (Path(path).stem, [float(x) for x in vec])
    except Exception as exc:  # keep the job alive on a bad file
        return (Path(path).stem, None)


def run_spark_feature_pipeline(
    audio_glob: str,
    out_parquet: str,
    cfg: Config,
    platform: Platform,
) -> str:
    _preflight_java()   # clear error instead of JAVA_GATEWAY_EXITED

    # Ensure Spark's Python workers use THIS interpreter (the venv), not whatever
    # bare `python3` the JVM finds on PATH. Without this, a system python of a
    # different version raises [PYTHON_VERSION_MISMATCH]. setdefault respects any
    # value the user set explicitly.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    try:
        from pyspark.sql import SparkSession, Row
    except Exception as exc:
        raise ImportError(
            "pyspark is required for the distributed pipeline. "
            "Install with `pip install pyspark`."
        ) from exc

    if platform.is_windows and not _windows_spark_ready():
        print(
            "[spark] On Windows, Spark needs HADOOP_HOME and winutils.exe set. "
            "See README (Windows + Spark). Attempting to continue anyway..."
        )

    # Normalize a bare local out path to an absolute file:// URI. Without a scheme,
    # Spark resolves it against the cluster's default FS — and if the machine's
    # HADOOP_CONF_DIR sets fs.defaultFS=hdfs://..., a local path is wrongly sent to
    # HDFS (ConnectException localhost:9000). An explicit file:// URI always writes
    # locally regardless of the host's Hadoop config.
    from urllib.parse import urlparse
    scheme = urlparse(out_parquet).scheme
    # len<2 covers a bare path ('') AND a Windows drive letter ('c' from "C:\..."),
    # which urlparse otherwise reads as a scheme. Path.as_uri() yields
    # file:///C:/x/y, which Spark accepts on every OS.
    if len(scheme) < 2:
        out_parquet = Path(out_parquet).resolve().as_uri()

    paths: List[str] = sorted(glob.glob(audio_glob))
    if not paths:
        raise FileNotFoundError(f"No audio matched: {audio_glob}")

    spark = (
        SparkSession.builder
        .appName("MoodSyncFeaturePipeline")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    try:
        rdd = spark.sparkContext.parallelize(paths, numSlices=min(8, len(paths)))
        # mapPartitions-style distributed feature extraction.
        rows = (
            rdd.map(lambda p: _feature_row(p, cfg))
            .filter(lambda r: r[1] is not None)
            .map(lambda r: Row(song_id=r[0], features=r[1]))
        )
        df = spark.createDataFrame(rows)
        df.write.mode("overwrite").parquet(out_parquet)
        n = df.count()
        print(f"[spark] wrote {n} feature rows -> {out_parquet}")
    finally:
        spark.stop()
    return out_parquet

#!/usr/bin/env python3
"""Prepare the DEAM dataset and point config.yaml at it.

Two modes:

    # Register an existing local copy
    python scripts/prepare_deam.py --local-path /data/DEAM

    # Download it (large; see the note on DEAM_SOURCES below)
    python scripts/prepare_deam.py --download --dest data_store/deam

The script discovers the audio and annotation directories inside the dataset
root, validates them against what `moodsync/data/deam.py` actually expects (it
calls `load_deam` and requires real clips back), then writes the two paths into
`config.yaml`. It is idempotent: re-running with the same dataset is a no-op.

Cross-platform: pathlib + stdlib urllib/zipfile only, no shell commands.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

# Make `moodsync` importable when run as a plain script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moodsync.config import Config, load_config              # noqa: E402
from moodsync.data.deam import load_deam                      # noqa: E402
from moodsync.platform_utils import normalize_path, project_root  # noqa: E402

# Official DEAM distribution (University of Geneva CVML). The audio is Creative
# Commons licensed and served directly — no login or terms form. Verified
# reachable; sizes are from the server's Content-Length.
DEAM_BASE_URL = "https://cvml.unige.ch/databases/DEAM"
DEAM_SOURCES = {
    "DEAM_audio.zip": f"{DEAM_BASE_URL}/DEAM_audio.zip",              # ~1.3 GB
    "DEAM_Annotations.zip": f"{DEAM_BASE_URL}/DEAM_Annotations.zip",  # ~4.7 MB
}
# Not needed for training, listed for reference:
#   {DEAM_BASE_URL}/metadata.zip   (~0.3 MB)  genre/duration/tags
#   {DEAM_BASE_URL}/features.zip   (~601 MB)  precomputed openSMILE features

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# ---------------------------------------------------------------- discovery --
def _find_audio_dir(root: Path) -> Path | None:
    """Directory holding the most audio files (DEAM mirrors nest it differently)."""
    best, best_n = None, 0
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        n = sum(1 for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES)
        if n > best_n:
            best, best_n = d, n
    return best


def _find_annotations_dir(root: Path) -> Path | None:
    """Directory that *directly* contains valence.csv and arousal.csv.

    `load_deam` resolves them with `rglob`, so any ancestor would technically
    work — but pointing at the exact folder keeps config.yaml self-explanatory
    and avoids picking up a stray valence.csv elsewhere in the tree.
    """
    val = next(root.rglob("valence.csv"), None)
    aro = next(root.rglob("arousal.csv"), None)
    if val is None or aro is None:
        return None
    if val.parent == aro.parent:
        return val.parent
    # Split across folders (some mirrors do this): fall back to a common ancestor.
    for cand in sorted({val.parent, aro.parent, root}, key=lambda p: -len(p.parts)):
        if next(cand.rglob("valence.csv"), None) and next(cand.rglob("arousal.csv"), None):
            return cand
    return root


# ----------------------------------------------------------------- download --
def _download(dest: Path) -> None:
    import urllib.error
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    for name, url in DEAM_SOURCES.items():
        archive = dest / name
        if archive.exists():
            print(f"[deam] {name} already downloaded, skipping")
        else:
            print(f"[deam] downloading {name} ...")
            try:
                urllib.request.urlretrieve(url, archive)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                raise SystemExit(
                    f"\n[deam] Download of {name} failed: {exc}\n"
                    f"       URL: {url}\n\n"
                    "Download it by hand from\n"
                    "  https://cvml.unige.ch/databases/DEAM/\n"
                    f"unzip both archives into {dest}, then re-run:\n"
                    f"  python scripts/prepare_deam.py --local-path {dest}\n"
                ) from exc
        marker = dest / (archive.stem + ".extracted")
        if marker.exists():
            print(f"[deam] {name} already extracted, skipping")
            continue
        print(f"[deam] extracting {name} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        marker.touch()


# ------------------------------------------------------------ config update --
def _replace_value(text: str, key: str, value: str) -> tuple[str, bool]:
    """Set `key: "value"` in YAML text, preserving indentation and trailing comments.

    Deliberately a surgical text edit rather than yaml.safe_load + yaml.dump,
    which would strip every comment from config.yaml.
    """
    pattern = re.compile(
        rf'^(?P<head>\s*{re.escape(key)}\s*:\s*)'
        r'(?P<val>"[^"]*"|\'[^\']*\'|[^#\n]*?)'
        r'(?P<tail>\s*#.*)?$',
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        raise KeyError(f"key {key!r} not found in config")
    new_line = f'{match.group("head")}"{value}"{match.group("tail") or ""}'
    if match.group(0) == new_line:
        return text, False
    return text[:match.start()] + new_line + text[match.end():], True


def update_config(config_path: Path, audio_dir: Path, ann_dir: Path,
                  dry_run: bool = False) -> bool:
    """Write both dataset paths into config.yaml. Returns True if it changed."""
    text = config_path.read_text(encoding="utf-8")
    text, c1 = _replace_value(text, "deam_audio", audio_dir.as_posix())
    text, c2 = _replace_value(text, "deam_annotations", ann_dir.as_posix())
    changed = c1 or c2
    if changed and not dry_run:
        config_path.write_text(text, encoding="utf-8")
    return changed


# ------------------------------------------------------------------- verify --
def verify(audio_dir: Path, ann_dir: Path, max_songs: int = 5) -> int:
    """Validate using the real loader. Returns the number of clips it resolved."""
    probe = Config({"datasets": Config({
        "deam_audio": str(audio_dir),
        "deam_annotations": str(ann_dir),
    })})
    clips = load_deam(probe, max_songs=max_songs)
    return len(clips) if clips else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prepare DEAM for MoodSync.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-path", type=str,
                     help="Root of an existing DEAM copy.")
    src.add_argument("--download", action="store_true",
                     help="Download DEAM into --dest (best-effort; see --help notes).")
    ap.add_argument("--dest", type=str, default=None,
                    help="Download/extract target (default: data_store/deam).")
    ap.add_argument("--config", type=str, default=None,
                    help="config.yaml to update (default: repo root).")
    ap.add_argument("--audio-dir", type=str, default=None,
                    help="Override audio directory discovery.")
    ap.add_argument("--annotations-dir", type=str, default=None,
                    help="Override annotations directory discovery.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written without touching config.yaml.")
    args = ap.parse_args(argv)

    if args.download:
        root = normalize_path(args.dest or (project_root() / "data_store" / "deam"))
        _download(root)
    else:
        root = normalize_path(args.local_path)
        if not root.exists():
            print(f"[deam] path does not exist: {root}", file=sys.stderr)
            return 2

    audio_dir = normalize_path(args.audio_dir) if args.audio_dir else _find_audio_dir(root)
    ann_dir = (normalize_path(args.annotations_dir) if args.annotations_dir
               else _find_annotations_dir(root))

    if audio_dir is None:
        print(f"[deam] no audio files found under {root}", file=sys.stderr)
        return 2
    if ann_dir is None:
        print(f"[deam] valence.csv / arousal.csv not found under {root}", file=sys.stderr)
        return 2

    print(f"[deam] audio       -> {audio_dir}")
    print(f"[deam] annotations -> {ann_dir}")

    n = verify(audio_dir, ann_dir)
    if n == 0:
        print("[deam] the loader resolved 0 songs — audio filenames must match the\n"
              "       song_id column in valence.csv/arousal.csv. Not writing config.",
              file=sys.stderr)
        return 1
    print(f"[deam] verified: loader resolved {n} song(s) from this layout")

    config_path = normalize_path(args.config) if args.config else project_root() / "config.yaml"
    changed = update_config(config_path, audio_dir, ann_dir, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[deam] dry run — {'would update' if changed else 'already correct'}: {config_path}")
    else:
        print(f"[deam] {'updated' if changed else 'already correct'}: {config_path}")
    print("[deam] done. Train on real data with: python -m moodsync.cli train --os mac")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

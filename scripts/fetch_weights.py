"""Fetch or check the pretrained weights listed in envs/weights.yaml.

Usage (project venv):
  uv run python scripts/fetch_weights.py                    # fetch every default entry that is missing
  uv run python scripts/fetch_weights.py --only moge2 aliked-n16
  uv run python scripts/fetch_weights.py --check            # offline presence check, prints a table
  uv run python scripts/fetch_weights.py --check --verify   # also sha256 every file (reads ~7 GB)
  uv run python scripts/fetch_weights.py --hf-home cache/e13/hf --check

An entry counts as present if it is found in any search location, in this order:
  --hf-home / --torch-home (repeatable) -> the manifest's hf_home / torch_home ->
  $HF_HOME / $TORCH_HOME -> ~/.cache/huggingface / ~/.cache/torch.
Missing entries are downloaded into the first --hf-home / --torch-home given, else into the
manifest location (which is where the consuming code looks by default).

--check never touches the network and never imports torch or loads a model: HF entries are
resolved with huggingface_hub.try_to_load_from_cache (local cache only), URL checkpoints by
file presence + sha256 (they are small), torch.hub code by its hubconf.py.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "envs" / "weights.yaml"
_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def expand(p: str) -> Path:
    s = _VAR.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), p)
    path = Path(os.path.expanduser(s))
    return path if path.is_absolute() else ROOT / path


def dedupe(paths: list[Path]) -> list[Path]:
    seen, out = set(), []
    for p in paths:
        key = os.path.realpath(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def search_dirs(entry: dict, args: argparse.Namespace) -> list[Path]:
    if "path" in entry:  # fixed-location file: one place to look
        return [expand(entry["path"]).parent]
    if entry["kind"] == "hf":
        cli, field, env, default = args.hf_home, "hf_home", "HF_HOME", "~/.cache/huggingface"
    else:
        cli, field, env, default = args.torch_home, "torch_home", "TORCH_HOME", "~/.cache/torch"
    dirs = [expand(d) for d in cli] + [expand(entry[field])]
    if os.environ.get(env):
        dirs.append(expand(os.environ[env]))
    dirs.append(expand(default))
    return dedupe(dirs)


def home_field(entry: dict) -> Path:
    if "path" in entry:
        return expand(entry["path"]).parent
    return expand(entry["hf_home" if entry["kind"] == "hf" else "torch_home"])


def target_dir(entry: dict, args: argparse.Namespace) -> Path:
    if "path" in entry:
        return home_field(entry)
    cli = args.hf_home if entry["kind"] == "hf" else args.torch_home
    return expand(cli[0]) if cli else home_field(entry)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return ""


# ---------------------------------------------------------------- presence checks
def check_hf(entry: dict, home: Path, verify: bool) -> tuple[str, int]:
    from huggingface_hub import try_to_load_from_cache

    cache = home / "hub"
    total, notes = 0, []
    for fn in entry["files"]:
        p = try_to_load_from_cache(entry["repo_id"], fn, cache_dir=cache, revision=entry["revision"])
        if not isinstance(p, str) or not os.path.exists(p):
            return f"missing {fn}", 0
        total += os.path.getsize(p)
        if verify and os.path.islink(p):
            # For LFS files the blob name is the file's sha256 (64 hex); git-sha1 blobs are skipped.
            etag = Path(os.readlink(p)).name
            if re.fullmatch(r"[0-9a-f]{64}", etag) and sha256(Path(p)) != etag:
                return f"sha256 mismatch {fn}", total
    ref = cache / f"models--{entry['repo_id'].replace('/', '--')}" / "refs" / "main"
    if not ref.exists():
        notes.append("no refs/main (load with revision=)")
    elif ref.read_text().strip() != entry["revision"]:
        notes.append(f"refs/main={ref.read_text().strip()[:7]} != pin")
    return ("ok" + (" (sha256 ok)" if verify else "") + ("; " + "; ".join(notes) if notes else "")), total


def url_dest(entry: dict, home: Path) -> Path:
    return expand(entry["path"]) if "path" in entry else home / "hub" / "checkpoints" / entry["filename"]


def check_url(entry: dict, home: Path, verify: bool) -> tuple[str, int]:
    p = url_dest(entry, home)
    if not p.is_file():
        return "missing", 0
    if sha256(p) != entry["sha256"]:
        return "sha256 MISMATCH", p.stat().st_size
    return "ok (sha256 ok)", p.stat().st_size


def check_hub_repo(entry: dict, home: Path, verify: bool) -> tuple[str, int]:
    d = home / "hub" / entry["dir"]
    if not (d / "hubconf.py").is_file():
        return "missing", 0
    return "ok (ref not recorded by torch.hub)", sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


CHECKS = {"hf": check_hf, "url": check_url, "torch_hub_repo": check_hub_repo}


def locate(entry: dict, args: argparse.Namespace) -> tuple[str, int, Path | None, list[Path]]:
    dirs = search_dirs(entry, args)
    first_status = "missing"
    for d in dirs:
        status, size = CHECKS[entry["kind"]](entry, d, args.verify)
        if status.startswith("ok"):
            return status, size, d, dirs
        if first_status == "missing" and not status.startswith("missing"):
            first_status = f"{status} @ {d}"
    return first_status, 0, None, dirs


# ---------------------------------------------------------------- downloads
def fetch_hf(entry: dict, home: Path) -> None:
    from huggingface_hub import snapshot_download

    cache = home / "hub"
    snapshot_download(entry["repo_id"], revision=entry["revision"], cache_dir=cache,
                      allow_patterns=entry["files"])
    # A download by commit sha writes no ref; offline loads by repo id resolve refs/main.
    ref = cache / f"models--{entry['repo_id'].replace('/', '--')}" / "refs" / "main"
    if not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(entry["revision"])
    elif ref.read_text().strip() != entry["revision"]:
        print(f"  note: {ref} points to {ref.read_text().strip()}, not the pinned revision")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)


def fetch_url(entry: dict, home: Path) -> None:
    dest = url_dest(entry, home)
    _download(entry["url"], dest)
    got = sha256(dest)
    if got != entry["sha256"]:
        bad = dest.with_name(dest.name + ".sha-mismatch")
        dest.replace(bad)
        raise RuntimeError(f"{entry['name']}: sha256 {got} != pinned {entry['sha256']} (kept as {bad})")


def fetch_hub_repo(entry: dict, home: Path) -> None:
    hub = home / "hub"
    dest = hub / entry["dir"]
    hub.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=hub) as td:
        zpath = Path(td) / "repo.zip"
        _download(f"https://github.com/{entry['repo']}/zipball/{entry['ref']}", zpath)
        with zipfile.ZipFile(zpath) as z:
            top = z.namelist()[0].split("/")[0]
            z.extractall(td)
        (Path(td) / top).replace(dest)
    # torch.hub asks before running code from an untrusted repo; mark it trusted like torch does.
    trusted = hub / "trusted_list"
    owner_name = entry["repo"].replace("/", "_")
    lines = trusted.read_text().split() if trusted.exists() else []
    if owner_name not in lines:
        with open(trusted, "a") as f:
            f.write(owner_name + "\n")


FETCHES = {"hf": fetch_hf, "url": fetch_url, "torch_hub_repo": fetch_hub_repo}


# ---------------------------------------------------------------- main
def short_home(p: Path | None) -> str:
    if p is None:
        return "-"
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p).replace(str(Path.home()), "~")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="NAME", help="entries to act on (default: all default:true)")
    ap.add_argument("--hf-home", action="append", default=[], metavar="DIR",
                    help="extra HF_HOME to search first; downloads go here (repeatable)")
    ap.add_argument("--torch-home", action="append", default=[], metavar="DIR",
                    help="extra TORCH_HOME to search first; downloads go here (repeatable)")
    ap.add_argument("--check", action="store_true", help="offline presence check only")
    ap.add_argument("--verify", action="store_true", help="with --check: sha256 HF LFS files too")
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    if args.check:
        os.environ["HF_HUB_OFFLINE"] = "1"
    models = yaml.safe_load(Path(args.manifest).read_text())["models"]
    names = {m["name"] for m in models}
    if args.only:
        unknown = set(args.only) - names
        if unknown:
            ap.error(f"unknown model(s): {', '.join(sorted(unknown))}; known: {', '.join(sorted(names))}")
        selected = [m for m in models if m["name"] in args.only]
    else:
        selected = [m for m in models if m.get("default", True)]

    rows, failed = [], 0
    for m in selected:
        status, size, home, dirs = locate(m, args)
        if home is None and not args.check:
            dest = target_dir(m, args)
            print(f"fetching {m['name']} ({m['size']}) -> {short_home(dest)}", flush=True)
            try:
                FETCHES[m["kind"]](m, dest)
            except Exception as e:  # noqa: BLE001 - report every failure, continue with the rest
                print(f"  FAILED: {e}", file=sys.stderr)
            status, size, home, dirs = locate(m, args)
        if home is None:
            failed += 1
            status = f"{status.upper()} (searched {', '.join(short_home(d) for d in dirs)})"
        elif os.path.realpath(home) != os.path.realpath(home_field(m)):
            var = "HF_HOME" if m["kind"] == "hf" else "TORCH_HOME"
            status += f"; consumers need {var}={short_home(home)}"
        rows.append((m["name"], m["env"], m["used_by"].split(";")[0], human(size) if size else "-",
                     short_home(home), status))

    hdr = ("model", "env", "used by", "on disk", "location", "status")
    widths = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    widths[-1] = 0
    line = lambda r: " | ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()  # noqa: E731
    print(line(hdr))
    print("-+-".join("-" * max(w, 6) for w in widths))
    for r in rows:
        print(line(r))
    print(f"\n{len(rows) - failed}/{len(rows)} present" + (" (offline check)" if args.check else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

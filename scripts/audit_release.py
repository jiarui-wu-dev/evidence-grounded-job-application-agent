"""Check the public file set without publishing or echoing matched values."""

import json
import re
import subprocess
import tempfile
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
DANGEROUS = re.compile(
    r"(^|/)(?:\.env(?!\.example(?:/|$))[^/]*|\.venv|node_modules|\.cache|__pycache__)(?:/|$)"
    r"|^(?:data/(?:candidate_profile|jobs)|qdrant_data|output|evaluation/(?:checkpoints|fixtures|results))/"
    r"|^evaluation/(?:multi_jd_gold|retrieval_(?:challenge|eval|holdout))\.json$"
    r"|\.(?:sqlite\w*|db(?:-\w+)?|worker(?:-\w+)?|pem|key|pfx|bak|zip|onnx|safetensors)$",
    re.IGNORECASE,
)
PATTERNS = {
    "provider credential": re.compile(rb"(?:sk-or-v1-|ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{20,}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    "personal path": re.compile(rb"(?:[A-Z]:[\\/]+Users[\\/]+[^\s/'\"<>]+|/home/[^\s/'\"<>]+/(?:projects|\.codex))", re.I),
}
DEMO_PRIVATE = re.compile(rb"[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}", re.I)
PUBLIC_FIXTURES = {
    "frontend/src/demo.json",
    "tests/fixtures/public_multi_jd_gold.json",
    "tests/fixtures/synthetic_targeted_revalidation.json",
}


def git(*arguments, root=ROOT, check=True):
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=check)


def publication_paths():
    if git("rev-parse", "--show-toplevel", check=False).returncode == 0:
        result = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    else:
        with tempfile.TemporaryDirectory() as directory:
            git("init", "--bare", "--quiet", directory)
            result = git(
                f"--git-dir={directory}", f"--work-tree={ROOT}", "-c", "core.bare=false",
                "ls-files", "--others", "--exclude-standard", "-z",
            )
    return sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})


def main():
    files = publication_paths()
    if not files or ".env.example" not in files or "frontend/src/demo.json" not in files:
        raise SystemExit("Audit could not establish the intended publication set.")
    errors = []
    alias_path = ROOT / "data/candidate_profile/demo_aliases.json"
    private_identifiers = (
        {identifier.casefold() for identifier in json.loads(alias_path.read_text(encoding="utf-8"))}
        if alias_path.is_file() else set()
    )
    private_values = tuple(
        value.encode("utf-8")
        for key, value in dotenv_values(ROOT / ".env", interpolate=False).items()
        if value and len(value) >= 12 and any(token in key.upper() for token in ("KEY", "TOKEN", "PASSWORD", "SECRET"))
    )
    for name in files:
        if DANGEROUS.search(name):
            errors.append(f"{name}: private or generated artifact")
            continue
        path = ROOT / name
        if path.is_symlink():
            errors.append(f"{name}: symlink requires manual review")
            continue
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if any(value in raw for value in private_values):
            errors.append(f"{name}: contains a configured credential")
        for label, pattern in PATTERNS.items():
            if pattern.search(raw):
                errors.append(f"{name}: possible {label}")
        if name in PUBLIC_FIXTURES:
            content = raw.decode("utf-8").casefold()
            if DEMO_PRIVATE.search(raw) or any(identifier in content for identifier in private_identifiers):
                errors.append(f"{name}: unredacted candidate identifier")
        if name == ".env.example" and not re.search(rb"(?m)^OPENROUTER_API_KEY=\r?$", raw):
            errors.append(f"{name}: API key example must be blank")
    for error in errors:
        print(error)
    print(f"Release audit checked {len(files)} paths. Findings: {len(errors)}.")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()

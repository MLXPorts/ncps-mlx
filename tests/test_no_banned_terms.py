import pathlib


def test_no_banned_terms_in_code():
    roots = [pathlib.Path("ncps"), pathlib.Path("examples"), pathlib.Path("tests")]
    banned = {"imitator", "shim", "mock", "fallback"}
    exts = {".py", ".md", ".rst", ".toml", ".yml", ".yaml", ".ini"}

    violations = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in exts:
                # Ignore this test file itself
                if path.name == "test_no_banned_terms.py":
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                lowered = text.lower()
                for term in banned:
                    if term in lowered:
                        violations.append((str(path), term))

    assert not violations, f"Banned terms found: {violations}"

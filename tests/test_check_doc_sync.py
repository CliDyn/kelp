import importlib.util
import pathlib

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "docs" / "check_doc_sync.py")
_spec = importlib.util.spec_from_file_location("check_doc_sync", _SCRIPT)
cds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cds)


def _write_map(tmp_path, body):
    p = tmp_path / "doc_map.yaml"
    p.write_text(body)
    return p


def test_load_map_parses_entries(tmp_path):
    p = _write_map(tmp_path, """
localization:
  doc: docs/localization.md
  code:
    - src/kelp/core/mesh.py
    - src/kelp/numerics/interpolate.py:_interp_leastsq_core
  verified_at: abc1234
""")
    m = cds.load_map(p)
    assert set(m) == {"localization"}
    assert m["localization"]["doc"] == "docs/localization.md"
    assert m["localization"]["verified_at"] == "abc1234"
    assert len(m["localization"]["code"]) == 2


def test_validate_structure_flags_missing_doc_and_code(tmp_path):
    m = {"x": {"doc": "docs/x.md",
               "code": ["src/missing.py"],
               "verified_at": "abc1234"}}
    problems = cds.validate_structure(m, root=tmp_path)
    joined = " ".join(problems)
    assert "docs/x.md" in joined
    assert "src/missing.py" in joined
    assert len(problems) == 2


def test_validate_structure_passes_when_present(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("# x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def foo():\n    pass\n")
    m = {"x": {"doc": "docs/x.md",
               "code": ["src/a.py"],
               "verified_at": "abc1234"}}
    assert cds.validate_structure(m, root=tmp_path) == []


def test_check_anchors_detects_missing_symbol(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(
        "def present():\n    pass\n")
    m = {"x": {"doc": "docs/x.md",
               "code": ["src/a.py:present", "src/a.py:absent"],
               "verified_at": "abc1234"}}
    problems = cds.check_anchors(m, root=tmp_path)
    assert len(problems) == 1
    assert "absent" in problems[0] and "src/a.py" in problems[0]


def test_check_anchors_accepts_class_and_def(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(
        "class Mesh:\n    pass\n\ndef build_search(m):\n    return m\n")
    m = {"x": {"doc": "docs/x.md",
               "code": ["src/a.py:Mesh", "src/a.py:build_search"],
               "verified_at": "abc1234"}}
    assert cds.check_anchors(m, root=tmp_path) == []


def test_find_drift_uses_injected_counter():
    m = {"loc": {"doc": "docs/loc.md",
                 "code": ["src/a.py", "src/b.py:Sym"],
                 "verified_at": "base"}}

    def fake_commits_since(path, ref):
        return ["c1 changed"] if path == "src/a.py" else []

    drift = cds.find_drift(m, commits_since=fake_commits_since)
    assert len(drift) == 1
    assert drift[0].subsystem == "loc"
    assert drift[0].path == "src/a.py"
    assert drift[0].commits == ["c1 changed"]


def test_find_drift_empty_when_nothing_changed():
    m = {"loc": {"doc": "d", "code": ["src/a.py"], "verified_at": "base"}}
    drift = cds.find_drift(m, commits_since=lambda p, r: [])
    assert drift == []


def test_bless_sets_verified_at_for_one_subsystem():
    m = {"a": {"doc": "da", "code": ["x"], "verified_at": "old"},
         "b": {"doc": "db", "code": ["y"], "verified_at": "old"}}
    out = cds.bless(m, "a", "newsha")
    assert out["a"]["verified_at"] == "newsha"
    assert out["b"]["verified_at"] == "old"


def test_bless_unknown_subsystem_raises():
    m = {"a": {"doc": "da", "code": ["x"], "verified_at": "old"}}
    with pytest.raises(KeyError):
        cds.bless(m, "nope", "newsha")


def test_commits_since_real_git(tmp_path):
    import subprocess as sp
    def git(*args):
        sp.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "a.py").write_text("v1\n")
    (tmp_path / "b.py").write_text("v1\n")
    git("add", "."); git("commit", "-qm", "base")
    base = sp.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                  capture_output=True, text=True).stdout.strip()
    (tmp_path / "a.py").write_text("v2\n")
    git("add", "."); git("commit", "-qm", "change a")
    changed = cds.commits_since("a.py", base, root=tmp_path)
    unchanged = cds.commits_since("b.py", base, root=tmp_path)
    assert len(changed) == 1 and "change a" in changed[0]
    assert unchanged == []


def test_run_report_returns_nonzero_on_problems(tmp_path, capsys):
    (tmp_path / "docs").mkdir()
    p = tmp_path / "docs" / "doc_map.yaml"
    p.write_text("""
x:
  doc: docs/missing.md
  code: [src/missing.py]
  verified_at: HEAD
""")
    rc = cds.run_report(map_path=p, root=tmp_path)
    assert rc != 0
    assert "doc not found" in capsys.readouterr().out


def test_run_report_clean_returns_zero(tmp_path, capsys):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("# x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def foo():\n    pass\n")
    p = tmp_path / "docs" / "doc_map.yaml"
    p.write_text("""
x:
  doc: docs/x.md
  code: [src/a.py:foo]
  verified_at: HEAD
""")
    rc = cds.run_report(map_path=p, root=tmp_path,
                        commits_since=lambda path, ref: [])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_validate_structure_flags_missing_verified_at(tmp_path):
    # An entry without verified_at must be flagged — otherwise find_drift
    # cannot key its git query and drift goes silently undetected.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("# x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def foo():\n    pass\n")
    m = {"x": {"doc": "docs/x.md", "code": ["src/a.py:foo"]}}   # no verified_at
    assert cds.validate_structure(m, root=tmp_path) == ["[x] missing verified_at"]


def test_run_report_notes_unverifiable_stamp(tmp_path, capsys):
    # A verified_at that does not resolve in THIS repository (a fresh-history
    # clone, a squashed export) makes `git log ref..HEAD` fail, and the drift
    # query then silently finds nothing.  The report must say so instead of
    # printing a bare OK.  The literal "HEAD" is the same trap: HEAD..HEAD is
    # always empty.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("# x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def foo():\n    pass\n")
    p = tmp_path / "docs" / "doc_map.yaml"
    p.write_text("""
x:
  doc: docs/x.md
  code: [src/a.py:foo]
  verified_at: deadbee
""")
    rc = cds.run_report(map_path=p, root=tmp_path,
                        commits_since=lambda path, ref: [])
    out = capsys.readouterr().out
    assert rc == 0                      # a note, not a failure
    assert "NOTE" in out and "deadbee" in out


def test_ref_resolves_true_for_head_in_a_real_repo(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.st"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "f").write_text("x")
    subprocess.run(["git", "add", "f"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "c"], cwd=tmp_path, check=True)
    assert cds.ref_resolves("HEAD", root=tmp_path) is True
    assert cds.ref_resolves("deadbee", root=tmp_path) is False


def _repo_with_drift(repo):
    """A git repo whose doc map is verified at `base`, with a later commit
    changing the anchored file."""
    import subprocess as sp
    def git(*args):
        sp.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "docs").mkdir()
    (repo / "docs" / "x.md").write_text("# x")
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("def foo():\n    pass\n")
    git("add", "."); git("commit", "-qm", "base")
    base = sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                  capture_output=True, text=True).stdout.strip()
    (repo / "src" / "a.py").write_text("def foo():\n    return 1\n")
    git("add", "."); git("commit", "-qm", "change a")
    p = repo / "docs" / "doc_map.yaml"
    p.write_text(f"""
x:
  doc: docs/x.md
  code: [src/a.py:foo]
  verified_at: {base}
""")
    return p


def test_run_report_finds_drift_from_any_working_directory(tmp_path, monkeypatch, capsys):
    """Audit 2026-10-04 (F-062): the drift query ran `git log` in the CURRENT
    directory, so the checker reported 'all docs in sync' whenever it was
    started outside the repository root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    p = _repo_with_drift(repo)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    rc = cds.run_report(map_path=p, root=repo)
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "DRIFT" in out and "src/a.py" in out


def test_bless_stamps_the_repository_head_not_the_cwd(tmp_path, monkeypatch):
    import subprocess as sp
    repo = tmp_path / "repo"
    repo.mkdir()
    p = _repo_with_drift(repo)
    head = sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                  capture_output=True, text=True).stdout.strip()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert cds.main(["--map", str(p), "--root", str(repo), "--bless", "x"]) == 0
    assert cds.load_map(p)["x"]["verified_at"] == head


def test_find_drift_queries_the_given_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    p = _repo_with_drift(repo)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    drift = cds.find_drift(cds.load_map(p), root=repo)
    assert [d.path for d in drift] == ["src/a.py"]

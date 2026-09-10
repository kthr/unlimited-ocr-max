"""The tokenizer resolves its files from a local directory OR a Hub repo id.

The Hub-id branch is the one that broke serving from the default `--model
<hub id>` (a local-only filesystem check turned the repo id into a bogus
relative path). These tests pin both branches so it cannot silently regress.
"""

from __future__ import annotations

import pytest

from unlimited_ocr_max import tokenizer as tk


def test_resolve_local_dir(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    tj, tc = tk._resolve_tokenizer_files(tmp_path, revision=None)
    assert tj == tmp_path / "tokenizer.json"
    assert tc == tmp_path / "tokenizer_config.json"


def test_resolve_local_dir_missing_tokenizer(tmp_path):
    with pytest.raises(FileNotFoundError, match="tokenizer.json is required"):
        tk._resolve_tokenizer_files(tmp_path, revision=None)


def test_resolve_hub_id_downloads_with_revision(tmp_path, monkeypatch):
    fake_tj = tmp_path / "cache_tokenizer.json"; fake_tj.write_text("{}")
    fake_tc = tmp_path / "cache_tokenizer_config.json"; fake_tc.write_text("{}")
    calls = []

    def fake_download(repo_id, filename, revision=None):
        calls.append((repo_id, filename, revision))
        return str(fake_tj if filename == "tokenizer.json" else fake_tc)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    tj, tc = tk._resolve_tokenizer_files("kthierbach/unlimited-ocr-max", revision="v0.2.1")
    assert tj == fake_tj and tc == fake_tc
    assert ("kthierbach/unlimited-ocr-max", "tokenizer.json", "v0.2.1") in calls
    assert ("kthierbach/unlimited-ocr-max", "tokenizer_config.json", "v0.2.1") in calls

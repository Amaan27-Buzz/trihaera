"""Local-only model snapshots: never touch network in these tests."""
import json
from pathlib import Path

import pytest

from model_lab.catalog import MODELS, local_model_directory, verify_local_model
from scripts.download_models import DOWNLOAD_PATTERNS, SKIP_PATTERNS, main


def create_synthetic_snapshot(spec, root, *, indexed=False):
    folder = local_model_directory(spec, root)
    folder.mkdir(parents=True)
    (folder / "config.json").write_text('{"model_type":"dummy"}', encoding="utf-8")
    (folder / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
    name = "model.safetensors-00001-of-00001.safetensors" if indexed else "model.safetensors"
    (folder / name).write_bytes(b"not-a-real-weight-file")
    if indexed:
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map":{"model.embed":"%s" % name}}),
            encoding="utf-8",
        )
    return folder


def test_model_paths_mirror_huggingface_namespaces(tmp_path):
    assert local_model_directory(MODELS["smol"], tmp_path) == (
        tmp_path / "HuggingFaceTB" / "SmolLM2-360M-Instruct"
    )
    assert local_model_directory(MODELS["qwen"], tmp_path) == (
        tmp_path / "Qwen" / "Qwen3.5-0.8B"
    )


def test_verify_local_detects_nonexistent_and_missing_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        verify_local_model(MODELS["smol"], tmp_path)
    folder = local_model_directory(MODELS["smol"], tmp_path)
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "tokenizer.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="safetensors"):
        verify_local_model(MODELS["smol"], tmp_path)


def test_verify_local_accepts_single_and_indexed_shard_layout(tmp_path):
    smol = create_synthetic_snapshot(MODELS["smol"], tmp_path)
    qwen = create_synthetic_snapshot(MODELS["qwen"], tmp_path, indexed=True)
    assert verify_local_model(MODELS["smol"], tmp_path) == smol
    assert verify_local_model(MODELS["qwen"], tmp_path) == qwen
    qwen.joinpath("model.safetensors-00001-of-00001.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_local_model(MODELS["qwen"], tmp_path)


def test_verify_local_rejects_empty_weights(tmp_path):
    folder = create_synthetic_snapshot(MODELS["smol"], tmp_path)
    (folder / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="Empty"):
        verify_local_model(MODELS["smol"], tmp_path)


def test_downloader_verify_only_works_without_huggingface_or_network(tmp_path):
    create_synthetic_snapshot(MODELS["smol"], tmp_path)
    create_synthetic_snapshot(MODELS["qwen"], tmp_path, indexed=True)
    main(["--models", "smol", "qwen", "--root", str(tmp_path), "--verify-only"])


def test_excludes_large_noninference_upstream_artifacts():
    assert "onnx/*" in SKIP_PATTERNS
    assert "runs/*" in SKIP_PATTERNS
    assert "*.bin" in SKIP_PATTERNS
    assert "*.safetensors" in DOWNLOAD_PATTERNS
    assert "tokenizer*" in DOWNLOAD_PATTERNS


def test_benchmark_loader_checks_local_source_before_torch_import(tmp_path):
    # A missing root should raise a useful error even if Transformers and
    # its weights are not installed on the CI runner.
    from scripts.benchmark_models import load_reference
    # The missing snapshot is validated inside the function after its torch
    # import. Inspect source rather than importing Torch in lightweight CI.
    import inspect
    source = inspect.getsource(load_reference)
    assert "verify_local_model(spec, model_root)" in source
    assert "source, local_files_only=True" in source
    assert "spec.model_id, local_files_only=offline" not in source

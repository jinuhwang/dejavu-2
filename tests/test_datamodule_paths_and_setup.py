import os
from pathlib import Path
import pytest
import hydra
from hydra.utils import instantiate


def _compose_cfg_for(dataset_group: str):
    # Compose using the repo configs, override dataset group and keep CPU trainer
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(
            config_name="train.yaml",
            overrides=[
                f"data={dataset_group}",
                "trainer=cpu",
                "logger=csv",
                # Use base model matching the available feature directory on the feature volume
                "model.net.base_model_name=openai/clip-vit-large-patch14",
            ],
        )
    return cfg


def test_nextqa_required_files_exist_or_skip():
    cfg = _compose_cfg_for("nextqa")
    # Expected files from current OmegaConf: use configured split paths directly
    required = [
        Path(str(cfg.paths.nextqa.splits.train)),
        Path(str(cfg.paths.nextqa.splits.val)),
        Path(str(cfg.paths.nextqa.splits.test)),
    ]
    missing = [str(p) for p in required if not Path(p).exists()]

    if missing and os.environ.get("REQUIRE_DATA_PATHS", "0") != "1":
        pytest.skip(f"Missing dataset CSVs: {missing}. Set REQUIRE_DATA_PATHS=1 to enforce.")

    # If enforcing, assert presence
    assert not missing, f"Missing dataset CSVs: {missing}"


def test_nextqa_datamodule_setup_smoke_or_helpful_error():
    cfg = _compose_cfg_for("nextqa")
    dm = instantiate(cfg.data)

    # Try setup; if files are missing, we expect a FileNotFoundError with guidance
    try:
        dm.setup("fit")
    except FileNotFoundError as e:
        msg = str(e)
        assert "NextQA split CSV not found at" in msg and "paths.data_dir" in msg
        # If user wants strictness, fail instead of accepting helpful error
        if os.environ.get("REQUIRE_DATA_PATHS", "0") == "1":
            raise
        pytest.skip("NextQA files missing; received helpful error as expected.")
    else:
        # Setup succeeded: train/val are instantiated
        assert dm.data_train is not None and dm.data_val is not None
        ds = dm.data_train
        n = len(ds)
        if n == 0:
            # Likely missing extracted features under feature/fps{}/train/...
            from src.utils.paths import get_path_manager
            pm = get_path_manager(cfg.paths)
            feat_dir = pm.get_feature_dir('nextqa', cfg.data.fps, cfg.data.train_split, dir_key='feature')
            msg = f"No samples found; expected extracted features under {feat_dir}."
            if os.environ.get("REQUIRE_FEATURES", "0") == "1":
                raise AssertionError(msg)
            pytest.skip(msg)
        # Probe a few samples across the dataset
        probe_idxs = sorted(set([0, max(0, n // 2), n - 1]))[:3]
        for idx in probe_idxs:
            item = ds[idx]
            # Expect a list/tuple like (frame_idx, i, compressed?, ref_one_hot, ref_mask)
            assert isinstance(item, (list, tuple)) and len(item) >= 2

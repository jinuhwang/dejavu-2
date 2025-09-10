#!/usr/bin/env python
import os
from typing import Optional

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # Ensure project root on sys.path
    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    from hydra.utils import instantiate, get_class
    from src.data.components.train import create_train_dataset

    regen_env = os.environ.get("REGENERATE_DATASET_INFO", "0").strip()
    regenerate = regen_env in {"1", "true", "True", "yes", "on"}

    # Prefer instantiating datasets directly to optionally control regeneration
    data_cfg = cfg.data

    # Common fields
    base_model_name = data_cfg.base_model_name
    fps = data_cfg.fps
    return_compressed = bool(data_cfg.get("return_compressed", False))
    use_start_end = bool(data_cfg.get("use_start_end", False))

    # Train dataset
    TrainCls = get_class(data_cfg.train_class)
    TrainDS = create_train_dataset(TrainCls)
    train_ds = TrainDS(
        pattern=list(data_cfg.train_pattern),
        split=data_cfg.train_split,
        base_model_name=base_model_name,
        fps=fps,
        step=int(data_cfg.train_step),
        use_start_end=use_start_end,
        return_compressed=return_compressed,
        reuse_dataset_info=not regenerate,
    )

    # Val dataset mirrors train creation
    ValCls = get_class(data_cfg.val_class)
    ValDS = create_train_dataset(ValCls)
    val_ds = ValDS(
        pattern=list(data_cfg.val_pattern),
        split=data_cfg.val_split,
        base_model_name=base_model_name,
        fps=fps,
        step=int(data_cfg.val_step),
        use_start_end=use_start_end,
        return_compressed=return_compressed,
        reuse_dataset_info=not regenerate,
    )

    # Test dataset (optional; use datamodule method for consistency of options)
    # If the group doesn’t define test fields, skip gracefully
    test_len: Optional[int] = None
    try:
        from src.data.components.test import create_test_dataset
        TestCls = get_class(data_cfg.test_class)
        TestDS = create_test_dataset(TestCls)
        test_ds = TestDS(
            split=data_cfg.test_split,
            base_model_name=base_model_name,
            fps=fps,
            refresh_interval=int(data_cfg.get("test_refresh_interval", 0)),
            is_sequential=bool(data_cfg.get("test_is_sequential", False)),
            return_compressed=return_compressed,
            reuse_dataset_info=not regenerate,
        )
        test_len = len(test_ds)
    except Exception:
        pass

    # Print concise report
    print("Data Inspection")
    print(f"  group: data={cfg.get('data', 'default')}")
    print(f"  base_model: {base_model_name}")
    print(f"  fps: {fps}")
    print(f"  return_compressed: {return_compressed}")
    print(f"  regenerate_info: {regenerate}")
    print(f"  train: {len(train_ds)} samples (split={data_cfg.train_split})")
    print(f"  val:   {len(val_ds)} samples (split={data_cfg.val_split})")
    if test_len is not None:
        print(f"  test:  {test_len} samples (split={data_cfg.test_split})")


if __name__ == "__main__":
    main()


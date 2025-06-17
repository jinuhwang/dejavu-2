from typing import Any, Dict, List, Tuple

import hydra
import torch
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import safetensors
from pathlib import Path

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    get_path_manager
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    path_manager = get_path_manager(cfg.paths)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule =hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")

    if cfg.ckpt_path:
        ckpt_path = Path(str(cfg.ckpt_path))
        if ckpt_path.is_dir():
            # If the given checkpoint is directory, and it contains a single file under it, pass it as ckpt_path
            ckpt_files = [f for f in ckpt_path.iterdir() if f.is_file() and f.name.startswith('epoch')]
            if len(ckpt_files) == 1:
                ckpt_path = ckpt_files[0]
            else:
                raise ValueError(f"Directory {ckpt_path} contains multiple files. Please specify a single checkpoint file.")
        if '.safetensors' in ckpt_path.name:
            state_dict = {}
            with safetensors.safe_open(ckpt_path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        elif '.ckpt' in ckpt_path.name:   
            state_dict = torch.load(ckpt_path, map_location="cpu")['state_dict']

        renamed_state_dict = {}

        if cfg.model.mode == 'original':  # Plain ReuseViT
            skip_keys = ['orig_model.', 'original_encoder_layer.']
            for key, value in state_dict.items():
                key = key.replace("net._orig_mod.", "")
                key = key.replace("net.blobnet.", "blobnet.")
                key = key.replace("net.model.", "model.")
                key = key.replace("reuse_module.", "")
                if any(skip_key in key for skip_key in skip_keys):
                    continue
                renamed_state_dict[key] = value
            ret = model.net.load_state_dict(renamed_state_dict, strict=False)
            assert len(ret.missing_keys) == 0, f"Missing keys: {ret.missing_keys}"
        elif cfg.model.mode in ['reuse-sequential', 'reuse-train']:
            # Testing sequential reuse is simpler because it uses training model
            for key, value in state_dict.items():
                key = key.replace("net._orig_mod.", "")
                renamed_state_dict[key] = value
            ret = model.net.load_state_dict(renamed_state_dict, strict=False)
            assert len(ret.missing_keys) == 0, f"Missing keys: {ret.missing_keys}"
        else:
            raise NotImplementedError(f"This mode shouldn't supply a checkpoint")

    trainer.test(model=model, datamodule=datamodule)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()

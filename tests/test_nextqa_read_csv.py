from pathlib import Path
import os
import importlib
import pandas as pd

from omegaconf import OmegaConf

from src.utils.paths import PathManager


def _set_path_manager_from_dir(base: Path):
    cfg = OmegaConf.load('configs/paths/data.yaml')
    cfg.root_dir = str(base)
    # replace global path manager used by read_csv
    import src.utils.paths as paths_mod
    paths_mod._path_manager = PathManager(cfg)


def _write_minimal_nextqa_files(base: Path, with_train=True, with_val=True, with_test=True):
    d = base / 'dataset' / 'nextqa'
    d.mkdir(parents=True, exist_ok=True)
    # map
    (d / 'map_vid_vidorID.json').write_text('{"vid123":"vidor001"}')
    # splits
    if with_train:
        (d / 'train.csv').write_text('video\nvid123\n')
    if with_val:
        (d / 'val.csv').write_text('video\nvid123\n')
    if with_test:
        (d / 'test.csv').write_text('video_id\nvid123\n')


def test_read_csv_succeeds_with_configured_paths(tmp_path):
    _write_minimal_nextqa_files(tmp_path)
    _set_path_manager_from_dir(tmp_path)

    # import after setting path manager cache
    nextqa = importlib.import_module('src.data.components.nextqa')
    df_train = nextqa.read_csv('train')
    df_val = nextqa.read_csv('val')
    df_test = nextqa.read_csv('test')

    assert 'video' in df_train.columns and len(df_train) == 1
    assert 'video' in df_val.columns and len(df_val) == 1
    # test split has video_id column but we add 'video'
    assert 'video' in df_test.columns and len(df_test) == 1
    # cleanup: reset global path manager
    import src.utils.paths as paths_mod
    paths_mod._path_manager = None


def test_read_csv_raises_helpful_error_on_missing_split(tmp_path):
    # Write only map, no splits
    _write_minimal_nextqa_files(tmp_path, with_train=False, with_val=False, with_test=False)
    _set_path_manager_from_dir(tmp_path)
    nextqa = importlib.import_module('src.data.components.nextqa')

    try:
        nextqa.read_csv('train')
        assert False, 'Expected FileNotFoundError'
    except FileNotFoundError as e:
        # error message should include the resolved path and guidance
        msg = str(e)
        assert 'NextQA split CSV not found at' in msg
        assert ('paths.data_dir' in msg) or ('paths.root_dir' in msg)
    finally:
        import src.utils.paths as paths_mod
        paths_mod._path_manager = None

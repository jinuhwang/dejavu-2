from pathlib import Path

from src.utils.paths import get_path_manager
import src.utils.paths as paths_mod


def test_nextqa_paths_default_data_dir():
    # ensure using fresh default config (not mutated by other tests)
    paths_mod._path_manager = None
    pm = get_path_manager(None)
    # data_dir should be set to the shared feature volume by default
    assert str(pm.paths['data_dir']).startswith('/mnt/raid/jwhwang/dejavu')

    # NextQA split CSVs should derive from data_dir
    assert pm.paths['nextqa']['splits']['train'] == Path(pm.paths['root_dir']) / 'dataset/nextqa/train.csv'
    assert pm.paths['nextqa']['splits']['val'] == Path(pm.paths['root_dir']) / 'dataset/nextqa/val.csv'
    assert pm.paths['nextqa']['splits']['test'] == Path(pm.paths['root_dir']) / 'dataset/nextqa/test.csv'

    # Map file path should also derive from data_dir
    assert pm.paths['nextqa']['map'] == Path(pm.paths['root_dir']) / 'dataset/nextqa/map_vid_vidorID.json'

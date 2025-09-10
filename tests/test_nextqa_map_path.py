from pathlib import Path

import os
from src.data.components.nextqa import _resolve_map_path
from src.utils.paths import get_path_manager


class DummyPM:
    def __init__(self, paths):
        self.paths = paths


def test_resolve_map_uses_configured_path(tmp_path):
    # Create a dummy map json
    mp = tmp_path / 'map_vid_vidorID.json'
    mp.write_text('{}')
    pm = DummyPM({'nextqa': {'map': mp}})
    assert _resolve_map_path(pm) == mp


def test_resolve_map_fallback_to_repo_copy():
    # Use real path manager but point configured path to something missing
    pm_real = get_path_manager(None)
    missing = Path('/this/path/does/not/exist/map_vid_vidorID.json')
    pm = DummyPM({
        'nextqa': {'map': missing},
    })
    p = _resolve_map_path(pm)
    # The fallback should be under PROJECT_ROOT/dataset/nextqa/...
    proj_root = os.environ.get('PROJECT_ROOT', '/workspace')
    assert str(p).startswith(os.path.join(proj_root, 'dataset/nextqa/'))

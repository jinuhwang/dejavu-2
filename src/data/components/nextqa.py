#!/usr/bin/env python
import warnings

from tqdm import tqdm

import pandas as pd
from pathlib import Path
import json

from datasets import load_dataset

from ...utils.dataset import count_available_features
from .base import BaseVideoDataset

import hydra
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from ...utils.paths import get_path_manager
from pathlib import Path

def _resolve_map_path(path_manager):
    """Return an existing map path, trying a repo fallback if needed.

    Prefers configured path (paths['nextqa']['map']). If missing, tries
    `${paths.root_dir}/dataset/nextqa/map_vid_vidorID.json`. Raises
    FileNotFoundError with both attempted paths if not found.
    """
    configured = path_manager.paths['nextqa']['map']
    if isinstance(configured, str):
        configured = Path(configured)
    if configured.exists():
        return configured

    # Fallback to repo dataset copy under PROJECT_ROOT (set by rootutils)
    import os
    proj_root = os.environ.get('PROJECT_ROOT', None)
    candidates = []
    if proj_root:
        candidates.append(Path(proj_root) / 'dataset' / 'nextqa' / 'map_vid_vidorID.json')
    # Also try a conventional workspace path
    candidates.append(Path('/workspace/dataset/nextqa/map_vid_vidorID.json'))
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"NextQA map file not found. Tried: {[str(p) for p in [configured]+candidates]}. "
        "Update configs/paths/data.yaml or set paths.data_dir to the correct base."
    )


def read_csv(split_or_path):
    path_manager = get_path_manager()

    if split_or_path in ['train', 'val', 'test']:
        path = path_manager.paths['nextqa']['splits'][split_or_path]
    else:
        path = split_or_path

    # load the map from vid to vidorID
    map_path = _resolve_map_path(path_manager)
    with open(map_path, 'r') as f:
        map_vid_vidorID = json.load(f)

    try:
        if split_or_path == 'test':
            df = pd.read_csv(
                path,
                dtype={
                    'video_id': str,
                }
            )
            df['video'] = df['video_id']
        else:
            df = pd.read_csv(
                path,
                dtype={
                    'video': str,
                }
            )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"NextQA split CSV not found at {path}. Set paths.data_dir to your datasets base "
            f"(currently {path_manager.paths.get('data_dir')}) or override configs/paths/data.yaml."
        ) from e

    # Add column as vidorID
    df['vidorID'] = df['video'].map(lambda x: map_vid_vidorID[x])

    return df

def get_video_paths(youtube_ids, fps, resolution=None):
    path_manager = get_path_manager()

    video_dir = path_manager.get_video_dir('nextqa', fps, resolution=resolution)
    video_paths = [video_dir / f'{youtube_id}.mp4' for youtube_id in youtube_ids]
    return video_paths

def get_youtube_ids_and_paths(split_or_path, fps, resolution=None, return_time=False):
    assert return_time == False, "NextQA does not have time information"

    df = read_csv(split_or_path)

    video_ids = df['video'].unique()

    # retrieve the vidorIDs corresponding to the video_ids
    vidor_ids = []
    for video_id in video_ids:
        vidor_ids.append(df[df['video'] == video_id]['vidorID'].values[0])
    video_paths = get_video_paths(vidor_ids, fps, resolution=resolution)
    start_ends = None

    return video_ids, video_paths, start_ends


# Build dataset from the frames
class NextqaDataset(BaseVideoDataset):
    def __init__(
            self,
            split,
            base_model_name,
            fps,
            return_pixel_values=False,
            return_input_values=True,
            return_hidden_states=False,
            return_output_states=False,
            return_compressed=False,
            reuse_dataset_info=True,
            use_start_end=False,
            dir_key='feature',
        ):

        assert use_start_end == False, "NextQA does not have start and end information"

        USE_FEATURE_PATH_V2 = True

        super().__init__(
            'nextqa',
            split,
            base_model_name,
            fps,
            return_pixel_values=return_pixel_values,
            return_input_values=return_input_values,
            return_hidden_states=return_hidden_states,
            return_output_states=return_output_states,
            return_compressed=return_compressed,
            reuse_dataset_info=reuse_dataset_info,
            use_start_end=use_start_end,
            use_feature_path_v2=USE_FEATURE_PATH_V2,
            dir_key=dir_key,
        )

        if self.dataset_info is None:
            print(f"Creating new info file: {self.dataset_info_path}")
            df = read_csv(split)

            # Count the number of features per video
            num_per_video = {}

            video_ids = df['video'].unique()

            count_info = count_available_features(
                dataset=self.dataset,
                fps=self.fps,
                split=self.split,
                base_model_name=self.base_model_name,
                video_ids=video_ids,
                check_compressed=return_compressed,
                use_feature_path_v2=USE_FEATURE_PATH_V2,
            )

            self.dataset_info = []    # youtube_id, frameidx
            for youtube_id, start, end, frame_cnt in count_info:
                if frame_cnt == 0:
                    warnings.warn(f'Video {youtube_id} has no features, it probably means error from sanitization')
                    continue

                for i in range(start, end):
                    self.dataset_info.append((youtube_id, i))

            self.save_dataset_info()

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
def main(cfg):
    from ...utils.paths import get_path_manager
    path_manager = get_path_manager(cfg.paths)

    # Resolve fields from existing config to avoid struct errors when keys are missing at root.
    # Prefer data.* and model.* sources, fall back to root if provided.
    def pick(*candidates, default=None):
        for c in candidates:
            try:
                # OmegaConf provides .get; attribute access may raise on struct
                if isinstance(c, tuple):
                    node, key = c
                    val = node.get(key, None)
                else:
                    val = c
                if val is not None:
                    return val
            except Exception:
                continue
        return default

    split = pick(cfg.get("split", None), cfg.data.get("split", None), cfg.data.get("val_split", None), default="val")
    base_model_name = pick(cfg.get("base_model_name", None), cfg.data.get("base_model_name", None), cfg.model.net.base_model_name)
    fps = pick(cfg.get("fps", None), cfg.data.get("fps", None))
    return_compressed = bool(pick(cfg.get("return_compressed", None), cfg.data.get("return_compressed", None), default=False))
    regenerate = bool(cfg.get("regenerate_dataset_info", False))

    TEST_STEP = 500

    print(f"NextQA dataset build — split={split}, base_model={base_model_name}, fps={fps}, return_compressed={return_compressed}, regenerate={regenerate}")

    dataset = NextqaDataset(
        split=split,
        base_model_name=base_model_name,
        fps=fps,
        return_compressed=return_compressed,
        reuse_dataset_info=not regenerate,
    )

    print(f'len: {len(dataset)}')

    for i in tqdm(range(0, len(dataset), TEST_STEP), desc='Testing'):
        _ = dataset[i]


if __name__ == '__main__':
    main()

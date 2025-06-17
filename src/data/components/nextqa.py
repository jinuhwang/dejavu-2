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

def read_csv(split_or_path):
    path_manager = get_path_manager()

    if split_or_path in ['train', 'val', 'test']:
        path = path_manager.paths['nextqa']['splits'][split_or_path]
    else:
        path = split_or_path

    # load the map from vid to vidorID
    with open(path_manager.paths['nextqa']['map'], 'r') as f:
        map_vid_vidorID = json.load(f)

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

    TEST_STEP = 500

    dataset = NextqaDataset(
        split=cfg.split,
        base_model_name=cfg.base_model_name,
        fps=cfg.fps,
        return_compressed=cfg.return_compressed,
        reuse_dataset_info=not cfg.regenerate_dataset_info,
    )

    print(f'len: {len(dataset)}')

    for i in tqdm(range(0, len(dataset), TEST_STEP), desc='Testing'):
        _ = dataset[i]


if __name__ == '__main__':
    main()
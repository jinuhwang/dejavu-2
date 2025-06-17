#!/usr/bin/env python
import warnings

from tqdm import tqdm

import pandas as pd
from pathlib import Path
import numpy as np

from datasets import load_dataset

from ...utils.dataset import count_available_features
from .base import BaseVideoDataset

import hydra
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from ...utils.paths import get_path_manager

def read_csv(split_or_path):
    path_manager = get_path_manager()

    if split_or_path in ['train', 'test']:
        df = pd.read_csv(
            path_manager.paths['how2qa']['splits'][split_or_path],
            names=['id', 'interval', 'wrong1', 'wrong2', 'wrong3', 'question', 'correct', 'query_id' ]
        )
        start_times = []
        end_times = []
        for interval in df['interval']:
            start, end = interval.strip('[]').split(':')
            start_times.append(float(start))
            end_times.append(float(end))
        df['start'] = start_times
        df['end'] = end_times
    elif split_or_path == 'frozenbilm':
        df = pd.read_csv(path_manager.paths['how2qa']['splits'][split_or_path])
        youtube_ids = []
        base_times = []
        for video_id in df['video_id']:
            splitted = video_id.split('_')
            youtube_ids.append('_'.join(splitted[:-2]))
            base_times.append(int(splitted[-2]))
        df['id'] = youtube_ids
        # FrozenBiLM has special way of formatting start and end time
        df['start'] += base_times
        df['end'] += base_times
    else:
        df = pd.read_csv(split_or_path)
    return df

def get_video_paths(youtube_ids, fps, resolution=None, base_model_name=None):
    assert base_model_name is None, "base_model_name is not supported for how2qa"
    path_manager = get_path_manager()

    video_dir = path_manager.get_video_dir('how2qa', fps, resolution=resolution)
    video_paths = [video_dir / f'{youtube_id}.mp4' for youtube_id in youtube_ids]
    return video_paths

def get_cropped_video_path(video_path, start, end):
    video_path = Path(video_path)
    return video_path.parent / f'{video_path.stem}_{start}_{end}.mp4'

def get_youtube_ids_and_paths(split_or_path, fps, resolution=None, base_model_name=None, return_time=False):
    df = read_csv(split_or_path)
    if not return_time:
        youtube_ids = df['id'].unique()
        start_ends = None
        video_paths = get_video_paths(youtube_ids, fps, resolution=resolution, base_model_name=base_model_name)
    else:
        id_start_end_list = list(set(
            (row['id'], row['start'], row['end'])
            for _, row in df.iterrows()
        ))
        youtube_ids, starts, ends = zip(*id_start_end_list)
        starts = [int(start) for start in starts]
        ends = [int(end) for end in ends]
        video_paths = get_video_paths(youtube_ids, fps, resolution=resolution, base_model_name=base_model_name)
        start_ends = list(zip(starts, ends))

        assert len(video_paths) == len(start_ends)

        if fps is not None:
            for idx, (video_path, start_end) in enumerate(zip(video_paths, start_ends)):
                start, end = start_end
                video_path = get_cropped_video_path(video_path, start, end)
                video_paths[idx] = video_path
    
    return youtube_ids, video_paths, start_ends


# Build dataset from the frames
class How2qaDataset(BaseVideoDataset):
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

        super().__init__(
            'how2qa',
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
            use_feature_path_v2=True,
            dir_key=dir_key,
        )

        if self.dataset_info is None:
            print(f"Creating new info file: {self.dataset_info_path}")
            df = read_csv(split)

            # Count the number of features per video
            num_per_video = {}

            if use_start_end:
                youtube_ids = df['id']
                starts = df['start'].apply(lambda x: int(x * fps))
                ends = df['end'].apply(lambda x: int(x * fps))
            else:
                youtube_ids = df['id'].unique()
                starts = None
                ends = None

            count_info = count_available_features(
                dataset=self.dataset,
                fps=self.fps,
                split=self.split,
                base_model_name=self.base_model_name,
                video_ids=youtube_ids,
                starts=starts,
                ends=ends,
                check_compressed=return_compressed,
                use_feature_path_v2=True,
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

    dataset = How2qaDataset(
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
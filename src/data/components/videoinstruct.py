#!/usr/bin/env python
import warnings

from .base import BaseVideoDataset
from ...utils.dataset import count_available_features, load_embedding
from ...utils.paths import get_path_manager
from tqdm import tqdm

import pandas as pd
from pathlib import Path
import numpy as np

from datasets import load_dataset

path_manager = get_path_manager()

def get_video_paths(video_ids, fps, resolution=None):
    video_dir = path_manager.get_video_dir('videoinstruct', fps, resolution=resolution)
    video_paths = [video_dir / f'{youtube_id}.mp4' for youtube_id in video_ids]
    return video_paths

def get_video_ids():
    ds = load_dataset("MBZUAI/VideoInstruct-100K", cache_dir=path_manager.paths['cache_dir'])

    video_ids = set()
    for row in ds['train']:
        video_ids.add(row['video_id'])
    video_ids = list(video_ids)
    return video_ids

def get_video_ids_and_paths(split_or_path, fps, base_model_name=None):
    assert split_or_path == 'train', "Only train split is supported for now"
    video_ids = get_video_ids()

    video_paths = get_video_paths(video_ids, fps, base_model_name=base_model_name)
    
    return video_ids, video_paths, None

# Build dataset from the frames
class VideoinstructDataset(BaseVideoDataset):
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

        assert split == 'train', "VideoInstruct only has train split"
        assert not use_start_end, "VideoInstruct does not use start and end"

        super().__init__(
            'videoinstruct',
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
            video_ids = get_video_ids()

            count_info = count_available_features(
                dataset=self.dataset,
                base_model_name=self.base_model_name,
                fps=self.fps,
                split=self.split,
                dir_key=self.dir_key,
                video_ids=video_ids,
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test videoinstruct dataset')
    parser.add_argument('--base-model-name', default='openai/clip-vit-base-patch16')
    parser.add_argument('--fps', type=float, default=1)
    parser.add_argument('--return-compressed', action='store_true', help='Return compressed features')
    parser.add_argument('--regenerate-dataset-info', action='store_true', help='Regenerate dataset info')
    args = parser.parse_args()

    TEST_STEP = 500

    dataset = VideoinstructDataset(
        split='train',
        base_model_name=args.base_model_name,
        fps=args.fps,
        return_compressed=args.return_compressed,
        reuse_dataset_info=not args.regenerate_dataset_info,
    )

    print(f'len: {len(dataset)}')

    for i in tqdm(range(0, len(dataset), TEST_STEP), desc='Testing'):
        _ = dataset[i]


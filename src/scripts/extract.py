#!/usr/bin/env python
import hydra
import argparse
from functools import partial
from pathlib import Path

import ray
from ray.util import ActorPool
import torch
from transformers import CLIPImageProcessor, SiglipImageProcessor, SiglipVisionModel
import numpy as np
import os
from tqdm import tqdm

import rootutils

from ..utils.dataset import get_resolution
from ..models.components.diffrate import create_diffrate_model
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class VideoProcessor:
    def __init__(
            self,
            get_feature_path_fn,
            processor_factory,
            model_factory,
            batch_size,
            device,
            dry_run=False,
            use_feature_v2=False,
            target_features=['i', 'p', 'o', 'h'],
        ):
        self.get_feature_path = get_feature_path_fn
        self.processor = None
        self.model = None
        self.device = torch.device(device)
        if device == 'cuda':
            gpu_id = ray.get_gpu_ids()[0]
            print(f'Using GPU {gpu_id}')
        self.batch_size = batch_size
        self.processor_factory = processor_factory
        self.model_factory = model_factory
        self.dry_run = dry_run
        self.use_feature_v2 = use_feature_v2
        self.target_features = target_features

    def get_processor(self):
        if self.processor is None:
            self.processor = self.processor_factory()
        return self.processor

    def get_model(self):
        if self.model is None:
            model= self.model_factory()
            self.model = model.to(self.device)
        return self.model

    def save_embedding(self, embedding, path):
        from ..utils.dataset import save_embedding

        if self.dry_run:
            print(f'DRYRUN: {path}')
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        save_embedding(embedding, path)

    def process_video(
            self,
            video_id,
            video_path,
            overwrite=False,
            start=None,
            end=None,
        ):
        from ..utils.dataset import get_all_frames

        if not Path(video_path).exists():
            print(f'Video does not exist: {video_path}')
            return

        frames = get_all_frames(video_path)

        if not frames:
            print('No frames available')
            return

        if start is None:
            start = 0
        if end is None:
            end = start + len(frames)

        if not overwrite:
            all_features_avaiable = self.check_all_features_available(video_id, start, end, self.target_features)
            if all_features_avaiable:
                print(f'All frames already processed for {video_id}')
                return

        # frames = frames[start:end]
        print(f'Processing {video_id} with {len(frames)} frames')

        processor = self.get_processor()

        pixel_values = processor(
            images=frames, return_tensors="pt", padding=True
        )["pixel_values"]

        if self.batch_size == -1:
            batch_size = len(pixel_values)
        else:
            batch_size = self.batch_size
            # At first, batch size should be 5 (including 0th frame)

        outer_batch_idx = 0
        while outer_batch_idx < len(pixel_values):
            batched_frames = frames[outer_batch_idx:outer_batch_idx+batch_size]
            batch_pixel_values = pixel_values[outer_batch_idx:outer_batch_idx+batch_size]
            batch_pixel_values_on_device = batch_pixel_values.to(self.device)

            if not self.target_features == ['i']:
                model = self.get_model()

                with torch.no_grad():
                    outputs = model(
                        pixel_values=batch_pixel_values_on_device,
                        output_hidden_states=True,
                    )
                    image_embeds = outputs.image_embeds.cpu()
                    hidden_states = outputs.hidden_states[-1].cpu()
                    second_hidden_states = outputs.hidden_states[-2].cpu()

            for inner_batch_idx in range(len(batch_pixel_values)):
                frame_idx = start + outer_batch_idx + inner_batch_idx

                i = batch_pixel_values[inner_batch_idx]
                i_path = self.get_feature_path(video_id, 'i', frame_idx, use_v2=self.use_feature_v2)

                if 'i' in self.target_features:
                    self.save_embedding(i, i_path)
                
                if 'p' in self.target_features:
                    p = batched_frames[inner_batch_idx]
                    p_path = self.get_feature_path(video_id, 'p', frame_idx, use_v2=self.use_feature_v2)
                    self.save_embedding(p, p_path)

                if 'o' in self.target_features:
                    o = image_embeds[inner_batch_idx]
                    o_path = self.get_feature_path(video_id, 'o', frame_idx, use_v2=self.use_feature_v2)
                    self.save_embedding(o, o_path)

                if 'h' in self.target_features:
                    h = hidden_states[inner_batch_idx]
                    h_path = self.get_feature_path(video_id, 'h', frame_idx, use_v2=self.use_feature_v2)
                    self.save_embedding(h, h_path)   

                if 'hh' in self.target_features:
                    hh = second_hidden_states[inner_batch_idx]
                    hh_path = self.get_feature_path(video_id, 'hh', frame_idx, use_v2=self.use_feature_v2)
                    hh_path.parent.mkdir(parents=True, exist_ok=True)
                    self.save_embedding(hh, hh_path)

            outer_batch_idx += batch_size

    def check_all_features_available(
        self,
        video_id,
        start,
        end,
        feature_types=['p', 'i', 'o', 'h'],
    ):
        from ..utils.dataset import load_embedding
        all_frames_loaded = True
        try:
            for frame_idx in range(start, end):
                # try to load from file
                for feature_type in feature_types:
                    feature_path = self.get_feature_path(video_id, feature_type, frame_idx, use_v2=self.use_feature_v2)
                    load_embedding(feature_path)

        except Exception as e:
            print(f'Error: {e}')
            all_frames_loaded = False

        return all_frames_loaded


def extract_features(
        youtube_ids,
        video_paths,
        get_feature_path_fn,
        processor_factory,
        model_factory,
        cfg,
        device,
        use_feature_v2,
        starts=None,
        ends=None,
        overwrite=False,
        target_features=['i', 'p', 'o', 'h'],
    ):
    num_workers = cfg.num_workers
    num_gpus = cfg.num_gpus
    batch_size = cfg.batch_size
    dry_run = cfg.dry_run

    def maybe_remote(device):
        if device == 'cpu':
            return ray.remote
        elif device == 'cuda':
            gpu_per_worker = num_gpus / num_workers
            return ray.remote(num_gpus=gpu_per_worker)
        else:
            raise NotImplementedError(f'Unknown device: {device}')

    DecoratedVideoProcessor = maybe_remote(device)(VideoProcessor)

    workers = [DecoratedVideoProcessor.remote(
        get_feature_path_fn,
        processor_factory,
        model_factory,
        batch_size,
        device,
        dry_run,
        use_feature_v2,
        target_features,
    ) for _ in range(num_workers)]
    pool = ActorPool(workers)

    ret = []

    def submit_to_actor(actor, idx):
        return actor.process_video.remote(
            youtube_ids[idx],
            video_paths[idx],
            overwrite=overwrite,
            start=None if starts is None else starts[idx],
            end=None if ends is None else ends[idx],
        )

    ret = pool.map_unordered(submit_to_actor, range(len(youtube_ids)))

    _ = [a for a in tqdm(ret, total=len(youtube_ids))]

@hydra.main(version_base="1.3", config_path="../../configs", config_name="extract.yaml")
def main(cfg):
    from ..utils.paths import get_path_manager
    path_manager = get_path_manager(cfg.paths)
    CACHE_DIR = path_manager.paths['cache_dir']

    from ..data.components import videoinstruct, how2qa, nextqa, msrvtt
    from ..models.components.clip import CLIPVisionModelWithProjection
    parser = argparse.ArgumentParser()

    target_features = cfg.target_features.split(',')
    print(f'Extracting features: {target_features}')

    if cfg.use_mixed_precision:
        torch.set_float32_matmul_precision('high')

    if cfg.extract_msrvtt_feature:
        assert cfg.mode == 'original', 'MSRVTT feature extraction is only supported for original mode'
        dir_key = 'msrvtt'
    elif cfg.mode == 'original':
        dir_key = 'feature'
    elif cfg.mode == 'diffrate':
        dir_key = 'diffrate'
    else:
        raise NotImplementedError
        
    if cfg.mode == 'original':
        get_feature_path_fn = partial(
            path_manager.get_feature_path,
            dataset=cfg.dataset,
            base_model_name=cfg.base_model_name,
            fps=cfg.fps,
            split=cfg.split,
            dir_key=dir_key
        )
    elif cfg.mode == 'diffrate':
        get_feature_path_fn = partial(
            path_manager.get_diffrate_path,
            dataset=cfg.dataset,
            base_model_name=cfg.base_model_name,
            fps=cfg.fps,
            split=cfg.split,
            dir_key=dir_key,
            flops=cfg.diffrate_flops,
        )

    if cfg.dataset == 'msrvtt' or cfg.extract_msrvtt_feature:
        name_or_checkpoint = path_manager.paths['msrvtt']['checkpoint']
        print(f'Extracting MSRVTT features from {name_or_checkpoint}')
    else:
        name_or_checkpoint = cfg.base_model_name

    def processor_factory(*_):
        if 'clip' in cfg.base_model_name:
            processor = CLIPImageProcessor.from_pretrained(
                cfg.base_model_name,
                cache_dir=CACHE_DIR
            )
        elif 'siglip' in cfg.base_model_name:
            processor = SiglipImageProcessor.from_pretrained(
                cfg.base_model_name,
                cache_dir=CACHE_DIR
            )
        return processor

    def model_factory(*_):
        if cfg.mode == 'original':
            if 'clip' in cfg.base_model_name:
                model = CLIPVisionModelWithProjection.from_pretrained(
                name_or_checkpoint,
                cache_dir=CACHE_DIR
            )
            elif 'siglip' in cfg.base_model_name:
                model = SiglipVisionModel.from_pretrained(
                    name_or_checkpoint,
                    cache_dir=CACHE_DIR
                )
        elif cfg.mode == 'diffrate':
            assert 'clip' in cfg.base_model_name, 'Diffrate is only supported for CLIP models'
            assert cfg.diffrate_flops is not None, 'Diffrate flops must be provided'
            
            model = create_diffrate_model(
                cfg.base_model_name,
                cfg.dataset,
                cfg.diffrate_flops,
            )

        if cfg.use_mixed_precision:
            torch.set_float32_matmul_precision('high')
        return model

    resolution = get_resolution(cfg.base_model_name)

    use_feature_v2 = False
    if cfg.dataset == 'how2qa':
        youtube_ids, video_paths, start_ends = how2qa.get_youtube_ids_and_paths(
            split_or_path=cfg.split,
            fps=cfg.fps,
            resolution=resolution,
            return_time=False
        )
        starts = None
        ends = None
        use_feature_v2 = True
    elif cfg.dataset == 'nextqa':
        youtube_ids, video_paths, start_ends = nextqa.get_youtube_ids_and_paths(
            split_or_path=cfg.split,
            fps=cfg.fps,
            resolution=resolution,
            return_time=False
        )
        starts = None
        ends = None
        use_feature_v2 = True
    elif cfg.dataset == 'msrvtt':
        youtube_ids, video_paths, _ = msrvtt.get_video_ids_and_paths(cfg.split, resolution=resolution, fps=cfg.fps)
        starts = None
        ends = None
        use_feature_v2 = True
    
    # elif cfg.dataset == 'kinetics':
    #     youtube_ids, video_paths, start_ends = kinetics.get_youtube_ids_and_paths(split=cfg.split, fps=cfg.fps)
    #     starts = [s * cfg.fps for s, _ in start_ends]
    #     ends = None
    # elif cfg.dataset == 'hmdb51':
    #     youtube_ids, video_paths, _ = hmdb51.get_video_names_and_paths(split=cfg.split, fps=cfg.fps)
    #     starts = None
    #     ends = None
    # elif cfg.dataset == 'activitynet':
    #     youtube_ids, video_paths, _ = activitynet.get_video_ids_and_paths(cfg.split, fps=cfg.fps)
    #     starts = None
    #     ends = None
    #     use_feature_v2 = True # Activitynet uses feature_v2
    elif cfg.dataset == 'videoinstruct':
        youtube_ids, video_paths, _ = videoinstruct.get_video_ids_and_paths(
            cfg.split,
            fps=cfg.fps,
            resolution=resolution,
        )
        starts = None
        ends = None
        use_feature_v2 = True # Videoinstruct uses feature_v2
    else:
        raise NotImplementedError

    # if cfg.world_size is not None:
    #     if cfg.rank is None:
    #         raise ValueError('Rank must be provided')
    #     world_size = cfg.world_size
    #     rank = cfg.rank

    #     if rank >= world_size:
    #         raise ValueError('Rank must be less than world size')

    #     youtube_ids, video_paths = split_per_rank(rank, world_size, youtube_ids, video_paths)


    if cfg.num_gpus > 0:
        device = 'cuda'
        ray.init(num_gpus=cfg.num_gpus)
    else:
        device = 'cpu'
        ray.init()

    extract_features(
        youtube_ids,
        video_paths,
        get_feature_path_fn,
        processor_factory,
        model_factory,
        cfg,
        device=device,
        use_feature_v2=use_feature_v2,
        starts=starts,
        ends=ends,
        overwrite=cfg.overwrite,
        target_features=target_features,
    )

if __name__ == '__main__':
    main()

import pandas as pd
import ray
from pathlib import Path
import subprocess
import shlex
import argparse
import subprocess
import hydra

import rootutils

from src.utils.dataset import get_resolution
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@ray.remote(num_cpus=4)
def run_transcode(input_path, output_path, log_path, resolution, fps, start_end=None, dry_run=False, overwrite=False):
    CMD = 'ffmpeg'
    if overwrite:
        CMD += ' -y'
    CMD += f" -i '{input_path}'"
    if start_end is not None:
        CMD += f' -ss {start_end[0]} -to {start_end[1]}'
    CMD += f' -vf "fps={fps},crop=\'min(iw,ih):min(iw,ih)\',scale={resolution}:{resolution}" -b_strategy 0 -bf 3'
    CMD += f' -c:v libx264 -preset slow -crf 22'
    CMD += f' -x264opts force-cfr:no-scenecut:subme=0:me=dia:ref=2'
    CMD += f' -threads 4'
    CMD += f' -an'
    CMD += f" '{output_path}'"

    stdout_path = log_path.with_suffix('.stdout')
    stderr_path = log_path.with_suffix('.stderr')

    print(CMD)
    if dry_run:
        return
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        handle = subprocess.Popen(
                shlex.split(CMD),
                stdout=stdout,
                stderr=stderr
                )
        handle.communicate()

@hydra.main(version_base="1.3", config_path="../../configs", config_name="transcode.yaml")
def main(cfg):
    from ..utils.paths import get_path_manager
    path_manager = get_path_manager(cfg.paths)
    
    from ..data.components import videoinstruct, how2qa, nextqa, msrvtt
    from ..utils.dataset import ray_get_with_tqdm, filter_existing_files, split_per_rank
    from ..utils.dataset import get_cropped_video_path

    start_ends = None
    if cfg.dataset == 'how2qa':
        youtube_ids, video_paths, start_ends = how2qa.get_youtube_ids_and_paths(
            split_or_path=cfg.split,
            fps=None,
            return_time=False
        )
    elif cfg.dataset == 'nextqa':
        youtube_ids, video_paths, start_ends = nextqa.get_youtube_ids_and_paths(
            split_or_path=cfg.split,
            fps=None,
            return_time=False
        )
    elif cfg.dataset == 'msrvtt':
        youtube_ids, video_paths, _ = msrvtt.get_video_ids_and_paths(cfg.split, fps=None)
    # elif cfg.DATASET == 'activitynet':
    #     youtube_ids, video_paths, _ = activitynet.get_video_ids_and_paths(cfg.SPLIT, fps=None)
    #     if cfg.base_model_name is None:
    #         print(f'Base model name is not provided. Assuming openai/clip-vit-large-patch14')
    #         width, height = 256, 256
    # elif cfg.DATASET == 'kinetics':
    #     youtube_ids, video_paths, _ = kinetics.get_youtube_ids_and_paths(split=cfg.SPLIT, fps=None)
    #     if cfg.base_model_name is None:
    #         print(f'Base model name is not provided. Assuming openai/clip-vit-base-patch16')
    #         width, height = 224, 224
    # elif cfg.DATASET == 'hmdb51':
    #     youtube_ids, video_paths, _ = hmdb51.get_video_names_and_paths(split=cfg.SPLIT, fps=None)
    #     if cfg.base_model_name is None:
    #         print(f'Base model name is not provided. Assuming openai/clip-vit-base-patch16')
    #         width, height = 224, 224
    # elif cfg.DATASET == 'activitynet':
    #     youtube_ids, video_paths, _ = activitynet.get_video_ids_and_paths(cfg.SPLIT, fps=None)
    #     if cfg.base_model_name is None:
    #         print(f'Base model name is not provided. Assuming openai/clip-vit-large-patch14')
    #         width, height = 256, 256
    elif cfg.dataset == 'videoinstruct':
        youtube_ids, video_paths, _ = videoinstruct.get_video_ids_and_paths(cfg.split, fps=None)
    else:
        raise NotImplementedError

    resolution = get_resolution(cfg.base_model_name)

    output_paths = []
    for idx, video_path in enumerate(video_paths):
        output_dir = path_manager.get_video_dir(cfg.dataset, cfg.fps, resolution)
        if cfg.dataset in ['hmdb51', 'nextqa']:
            output_dir = output_dir / video_path.parent.name
        output_path = output_dir / video_path.with_suffix('.mp4').name
        if start_ends is not None:
            start_end = start_ends[idx]
            output_path = get_cropped_video_path(output_path, start_end[0], start_end[1])
        output_paths.append(output_path)

    if not cfg.overwrite:
        output_paths, video_paths = filter_existing_files(output_paths, video_paths)

    tmp_output = []
    tmp_video = []
    for output_path, video_path in zip(output_paths, video_paths):
        if output_path in tmp_output:
            continue
        tmp_output.append(output_path)
        tmp_video.append(video_path)
    output_paths = tmp_output
    video_paths = tmp_video

    # if cfg.world_size is not None:
    #     if cfg.rank is None:
    #         raise ValueError('Rank must be provided')
    #     world_size = cfg.world_size
    #     rank = cfg.rank

    #     if rank >= world_size:
    #         raise ValueError('Rank must be less than world size')

    #     output_paths, video_paths = split_per_rank(rank, world_size, output_paths, video_paths)

    ret = []

    for idx in range(len(video_paths)):
        video_path = video_paths[idx]
        output_path = output_paths[idx]
        if start_ends is not None:
            start_end = start_ends[idx]
        else:
            start_end = None

        # Create output directory
        output_dir = Path(output_path).parent
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)


        log_path = log_dir / video_path.with_suffix('.log').name

        r = run_transcode.remote(
            video_path,
            output_path,
            log_path,
            resolution,
            cfg.fps,
            start_end=start_end,
            dry_run=cfg.dry_run,
            overwrite=cfg.overwrite
        )
        ret.append(r)

    ray_get_with_tqdm(ret)

if __name__ == '__main__':
    main()
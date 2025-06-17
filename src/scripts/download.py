import hydra
import ray
from pathlib import Path
import subprocess
import shlex
import subprocess
import json
from tqdm import tqdm
from pathlib import Path
import rootutils

from ..utils.dataset import ray_get_with_tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@ray.remote(num_cpus=1)
def run_download_pipeline(youtube_id, output_path, log_path, start_end=None, dry_run=False):
    url = f'https://www.youtube.com/watch?v={youtube_id}'
    # CMD = f"youtube-dl --format mp4 -o '{output_path}' '{url}'"
    CMD = f"yt-dlp -S ext:mp4"
    if start_end is not None:
        start, end = start_end
        CMD += f" --download-sections '*{start}-{end}'"
    CMD +=  f" -o '{output_path}' '{url}'"

    stdout_path = log_path.with_suffix('.stdout')
    stderr_path = log_path.with_suffix('.stderr')

    print(f'Running: {CMD}')
    if dry_run:
        return

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        handle = subprocess.Popen(
                shlex.split(CMD),
                stdout=stdout,
                stderr=stderr
                )
        handle.communicate()

def download_youtube_videos(youtube_ids, output_paths, start_ends=None, dry_run=False):
    # Create output directory
    output_dir = Path(output_paths[0]).parent
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    ret = []

    for idx in range(len(youtube_ids)):
        youtube_id = youtube_ids[idx]
        output_path = output_paths[idx]

        log_path = log_dir / f'{youtube_id}.log'

        if start_ends is not None:
            start_end = start_ends[idx]
        else:
            start_end = None

        r = run_download_pipeline.remote(youtube_id, output_path, log_path, start_end, dry_run=dry_run)
        ret.append(r)

    ray_get_with_tqdm(ret)

def is_video_decodable(video_path):
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'json',
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        data = json.loads(result.stdout)

        # If the 'duration' key exists in the output, the video is likely decodable.
        if "format" in data and "duration" in data["format"]:
            return True
    except subprocess.CalledProcessError:
        pass
    return False

@ray.remote(num_cpus=1)
def check_decodable(file_path):
    return file_path, is_video_decodable(file_path)

def delete_undecodable_files(directory):
    directory = Path(directory)
    total_items = sum(1 for _ in directory.iterdir())

    undecodable_files = []
    
    ret = []
    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue
        ret.append(check_decodable.remote(file_path))

    with tqdm(total=len(ret), desc="Checking decodability") as pbar:
        while ret:
            # Get results as they complete
            done_refs, ret = ray.wait(ret)
            for ref in done_refs:
                pbar.update(1)
                path, result = ray.get(ref)
                if not result:
                    undecodable_files.append(path)

    print(f"Out of {total_items} files, {len(undecodable_files)} files are undecodable.")

    while True:
        print("Press 'y' to delete, 'q' to quit:")
        key = input()
        if key == 'y':
            for file_path in undecodable_files:
                file_path.unlink()
            break
        elif key == 'q':
            break

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg):
    from ..utils.paths import get_path_manager
    path_manager = get_path_manager(cfg.paths)
    
    from ..data.components import videoinstruct, how2qa
    from ..utils.dataset import ray_get_with_tqdm, filter_existing_files


    if cfg.dataset == 'videoinstruct':
        print("You should download video files from Google drive link, follow the instructions:")
        print("https://github.com/mbzuai-oryx/Video-ChatGPT/issues/98")
        print(f"Please unzip them under the path: {path_manager.get_video_dir('videoinstruct')}")
        exit()
    elif cfg.dataset == 'how2qa':
        youtube_ids, video_paths, _ = how2qa.get_youtube_ids_and_paths(split_or_path=cfg.split, fps=None)
        start_ends = None
    # elif args.DATASET == 'kinetics':
    #     youtube_ids, video_paths, start_ends = kinetics.get_youtube_ids_and_paths(split=args.SPLIT)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} is not supported")

    # delete_undecodable_files(PREDEFINED_PATHS['how2qa']['video'])
    video_paths, youtube_ids = filter_existing_files(video_paths, youtube_ids)

    if len(youtube_ids) > 0:
        download_youtube_videos(youtube_ids, video_paths, start_ends, dry_run=cfg.dry_run)

if __name__ == '__main__':
    main()
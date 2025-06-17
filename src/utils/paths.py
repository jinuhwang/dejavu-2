from pathlib import Path
from typing import Dict, Any
import re
from omegaconf import OmegaConf

_path_manager = None

def rename_base_model(base_model_name):
    base_model_renamed = base_model_name.replace('/', '_')
    return base_model_renamed

def is_integer(value):
    try:
        int_value = int(value)
        float_value = float(value)
        if float_value == int_value:
            return True
        else:
            return False
    except ValueError:
        return False


class PathManager:
    def __init__(self, config):
        if config is None:
            print("Config is None, using default config")
            config = OmegaConf.load('/workspace/configs/paths/data.yaml')
        self.paths = OmegaConf.to_container(config, resolve=True)
        # Convert string paths to Path objects
        self._convert_to_path_objects(self.paths)
    
    def _convert_to_path_objects(self, config: Dict[str, Any]):
        for key, value in config.items():
            if isinstance(value, str) and ('/' in value or '\\' in value):
                config[key] = Path(value)
            elif isinstance(value, dict):
                self._convert_to_path_objects(value)
    
    def get_feature_dir(self, dataset: str, fps: int, split: str, dir_key: str = 'feature') -> Path:
        base_dir = self.paths[dataset][dir_key]

        # To refrain from naming file name 1.0
        if is_integer(fps):
            fps = int(fps)

        feature_dir = base_dir / f'fps{fps}' / split
        return feature_dir

    def get_video_dir(self, dataset: str, fps=None, resolution=None) -> Path:
        # If fps is None, return the original video directory
        if fps is None:
            assert resolution is None, "Resolution should not be provided when fps is None"
            return self.paths[dataset]['video']

        base_dir = self.paths[dataset]['video_transcoded']

        if resolution is not None:
            base_dir /= f"{resolution}"
        # To refrain from naming file name 1.0
        if is_integer(fps):
            fps = int(fps)
        ret = base_dir / f"{fps}fps"
        return ret

    def get_diffrate_path(
        self,
        video_id: str,
        feature_type: str,
        frame_num: int,
        dataset: str,
        fps: int,
        split: str,
        base_model_name: str = None,
        flops: float = None,
        dir_key: str = 'diffrate',
        use_v2: bool = False,
        resolution: str = None,
    ) -> Path:
        feature_dir = self.get_feature_dir(dataset, fps, split, dir_key)
        feature_dir /= f'{flops:.1f}'
        return self.append_feature_path(feature_dir, video_id, frame_num, feature_type, base_model_name, use_v2, resolution)

    def get_eventful_path(
        self,
        video_id: str,
        feature_type: str,
        frame_num: int,
        dataset: str,
        fps: int,
        split: str,
        base_model_name: str = None,
        topk: int = None,
        dir_key: str = 'eventful',
        use_v2: bool = False,
        resolution: str = None,
    ) -> Path:
        feature_dir = self.get_feature_dir(dataset, fps, split, dir_key)
        feature_dir /= f'{topk}'
        return self.append_feature_path(feature_dir, video_id, frame_num, feature_type, base_model_name, use_v2, resolution)

    def get_cmc_path(
        self,
        video_id: str,
        feature_type: str,
        frame_num: int,
        dataset: str,
        fps: int,
        split: str,
        base_model_name: str = None,
        threshold: float = None,
        dir_key: str = 'cmc',
        use_v2: bool = False,
    ) -> Path:
        feature_dir = self.get_feature_dir(dataset, fps, split, dir_key)
        feature_dir /= f'{threshold:.2f}'
        return self.append_feature_path(feature_dir, video_id, frame_num, feature_type, base_model_name, use_v2)

    def get_reuse_path(
        self,
        video_id: str,
        feature_type: str,
        frame_num: int,
        dataset: str,
        fps: int,
        split: str,
        base_model_name: str = None,
        reuse_model_name: str = None,
        dir_key: str = 'reuse',
        use_v2: bool = False,
        resolution: str = None,
    ) -> Path:
        feature_dir = self.get_feature_dir(dataset, fps, split, dir_key)
        feature_dir /= f'{reuse_model_name}'
        return self.append_feature_path(feature_dir, video_id, frame_num, feature_type, base_model_name, use_v2, resolution)

    def get_feature_path(
        self,
        video_id: str,
        feature_type: str,
        frame_num: int,
        dataset: str,
        fps: int,
        split: str,
        base_model_name: str = None,
        dir_key: str = 'feature',
        use_v2: bool = False,
        resolution: str = None,
    ) -> Path:
        feature_dir = self.get_feature_dir(dataset, fps, split, dir_key)
        return self.append_feature_path(feature_dir, video_id, frame_num, feature_type, base_model_name, use_v2, resolution)
        
    def append_feature_path(
        self,
        feature_dir,
        video_id,
        frame_num,
        feature_type,
        base_model_name,
        use_v2,
        resolution: str = None,
    ):
        assert feature_type in ['i', 'p', 'h', 'hh', 'o', 'c'], f"Invalid feature type {feature_type} (must be one of 'i', 'p', 'h', 'hh', 'o', 'c')"

        if feature_type in ['p', 'c']:
            if resolution is None:
                from src.utils.dataset import get_resolution
                resolution = get_resolution(base_model_name)
            feature_dir = feature_dir / f"{resolution}"
        else:
            renamed_base_model_name = rename_base_model(base_model_name)
            feature_dir /= renamed_base_model_name

        if use_v2:
            return feature_dir / f'{video_id}' / f'{feature_type}_{frame_num}.npz'
        else:
            return feature_dir / f'{video_id}_{feature_type}_{frame_num}.npz'

    def get_diffrate_prune_merge(self, dataset, flops, epoch=None):
        log_path = self.paths['diffrate_dir'] / f'{dataset}' / f'{flops:.1f}' / 'log_rank0.txt'

        with open(log_path, 'r') as f:
            for line in f:
                if 'INFO prune kept number:' in line:
                    prune_txt = line.strip()
                elif 'INFO merge kept number:' in line:
                    merge_txt = line.strip()
                if epoch is not None and f'Epoch: [{epoch}]' in line:
                    break

        # Parse the list from the following pattern
        # [2024-02-06 23:47:12 root] (engine.py 118): INFO prune kept number:[197, 193, 169, 155, 126, 106, 103, 98, 87, 73, 60, 5]
        # [2024-02-06 23:47:12 root] (engine.py 119): INFO merge kept number:[197, 175, 160, 148, 107, 103, 101, 91, 79, 65, 57, 5]
        # [2024-02-06 23:47:11 root] (utils.py 286): INFO Epoch: [299]  [259/260]
        pat = re.compile(r'.*\[(.*)\]')

        prune = pat.match(prune_txt).group(1)
        merge = pat.match(merge_txt).group(1)

        prune = list(map(int, prune.split(', ')))
        merge = list(map(int, merge.split(', ')))

        return prune, merge

def get_path_manager(cfg=None):
    global _path_manager

    if _path_manager is None:
        _path_manager = PathManager(cfg)

    return _path_manager
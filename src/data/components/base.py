import torch
from pathlib import Path
from joblib import dump, load
from ...utils.dataset import load_embedding
import torch
from pathlib import Path
import joblib
from typing import Optional, Dict, Any
from ...utils.paths import get_path_manager, rename_base_model
from functools import partial

class BaseVideoDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset: str,
            split: str,
            base_model_name: str,
            fps,
            return_pixel_values: bool = False,
            return_input_values: bool = True,
            return_hidden_states: bool = False,
            return_output_states: bool = False,
            return_compressed: bool = False,
            use_start_end: bool = False,
            use_feature_path_v2: bool = False,
            dir_key: str = 'feature',
            reuse_dataset_info: bool = True
        ):
        """Base Video Dataset implementation.
        
        Args:
            dataset: Name of the dataset (e.g., 'msrvtt', 'how2qa')
            split: Dataset split ('train', 'val', 'test')
            base_model_name: Name of the base model
            fps: Frames per second
            return_pixel_values: Whether to return raw pixel values
            return_input_values: Whether to return input embeddings
            return_hidden_states: Whether to return hidden states
            return_output_states: Whether to return output states
            return_compressed: Whether to return compressed features
            use_start_end: Whether to use start/end frame markers
        """
        self.dataset = dataset
        self.split = split
        self.base_model_name = base_model_name
        self.fps = fps
        self.dir_key = dir_key

        path_manager = get_path_manager()

        self.get_feature_path = partial(
            path_manager.get_feature_path,
            dataset=dataset,
            fps=fps,
            split=split,
            base_model_name=base_model_name,
            dir_key=dir_key,
            use_v2=use_feature_path_v2
        )
        
        # Set return flags
        self.return_pixel_values = return_pixel_values
        self.return_input_values = return_input_values
        self.return_hidden_states = return_hidden_states
        self.return_output_states = return_output_states
        self.return_compressed = return_compressed

        self.use_start_end = use_start_end
        self.use_feature_path_v2 = use_feature_path_v2
        self.reuse_dataset_info = reuse_dataset_info
        
        # Load dataset info
        self.dataset_info_path = self._get_dataset_info_path(return_compressed, use_start_end)
        self.dataset_info = self._load_dataset_info()
        
    def _get_dataset_info_path(self, return_compressed: bool, use_start_end: bool) -> Path:
        """Get path to dataset info file with appropriate suffixes."""
        path_manager = get_path_manager()

        feature_dir = path_manager.get_feature_dir(
            dataset=self.dataset,
            fps=self.fps,
            split=self.split,
            dir_key=self.dir_key
        )
        path = feature_dir / 'dataset_info.joblib'

        if return_compressed:
            renamed_base_model = rename_base_model(self.base_model_name)
            path = path.parent / renamed_base_model  / path.name
        if use_start_end:
            path = path.with_name(f"{path.stem}_start_end{path.suffix}")

        return path
    
    def _load_dataset_info(self) -> Dict[str, Any]:
        """Load dataset info from cache."""
        print(f'Loading dataset info from {self.dataset_info_path}')
        if self.reuse_dataset_info and self.dataset_info_path.exists():
            return joblib.load(self.dataset_info_path)
        else:
            print(f'Failed loading dataset info from {self.dataset_info_path}, child class should generate it')
            return None

    def save_dataset_info(self):
        self.dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.dataset_info, self.dataset_info_path)
    
    def __len__(self) -> int:
        return len(self.dataset_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find the video index
        youtube_id, frame_idx = self.dataset_info[idx]

        if not isinstance(frame_idx, torch.Tensor):
            frame_idx = torch.tensor(frame_idx)

        ret = (frame_idx,)
        if self.return_pixel_values:
            p_path = self.get_feature_path(youtube_id, 'p', frame_idx)
            ret += (load_embedding(p_path),)
        if self.return_input_values:
            i_path = self.get_feature_path(youtube_id, 'i', frame_idx)
            ret += (load_embedding(i_path),)
        if self.return_hidden_states:
            h_path = self.get_feature_path(youtube_id, 'h', frame_idx)
            ret += (load_embedding(h_path),)
        if self.return_output_states:
            o_path = self.get_feature_path(youtube_id, 'o', frame_idx)
            ret += (load_embedding(o_path),)
        if self.return_compressed:
            T = -3
            c_path = self.get_feature_path(youtube_id, 'c', frame_idx)
            cur_embedding = load_embedding(c_path)

            prev_embeddings = []
            for t in range(T, 0):
                if idx + t < 0:
                    prev_embeddings.append(torch.zeros_like(cur_embedding))
                    continue

                prev_id, prev_frame_idx = self.dataset_info[idx + t]

                if prev_id != youtube_id:
                    prev_embeddings.append(torch.zeros_like(cur_embedding))
                else:
                    prev_c_path = self.get_feature_path(prev_id, 'c', prev_frame_idx)
                    prev_embeddings.append(load_embedding(prev_c_path))

            prev_embeddings.append(cur_embedding)
            ret += (torch.stack(prev_embeddings, dim=1),)

        return ret

    def is_start_of_new_video(self, idx):
        _, frame_idx = self.dataset_info[idx]
        return frame_idx == 0
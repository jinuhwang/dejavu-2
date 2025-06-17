import hydra
import torch
import numpy as np
import warnings
import copy

def is_new_video(video_id, frame_idx, prev_video_id, prev_frame_idx):
    return video_id != prev_video_id or frame_idx == 0 or frame_idx != prev_frame_idx + 1

class ExtractionDatasetMixin:
    def __init__(self) -> None:

        # (id, start idx of current segment in dataset_info) => lenght of current segment
        self.id_idx2len = {}

        # Calculate 
        prev_frame_idx = -1
        prev_video_id = ''
        for idx, (video_id, frame_idx) in enumerate(self.dataset_info):
            if is_new_video(video_id, frame_idx, prev_video_id, prev_frame_idx):
                segment_start_idx = idx
                self.id_idx2len[(video_id, segment_start_idx)] = 0
            self.id_idx2len[(video_id, segment_start_idx)] += 1
            prev_frame_idx = frame_idx
            prev_video_id = video_id

        self.super_id_idx = []

        for (video_id, start_frame_idx), segment_len in self.id_idx2len.items():
            for i in range(segment_len):
                self.super_id_idx.append(((video_id, start_frame_idx), start_frame_idx + i))

    def __getitem__(self, idx):
        (video_id, _), super_idx = self.super_id_idx[idx]
        items = super().__getitem__(super_idx)

        return (video_id,) + tuple(items)

    def __len__(self):
        return len(self.super_id_idx)

class ReuseExtractionDatasetMixin:
    def __init__(self, num_frames, refresh_interval=0, is_sequential=False):
        assert num_frames == 4, 'Only support 4 frames for now'

        # TODO: deal with how2qa edge cases.
        # same video, several sequences

        super_id_idx = self.super_id_idx

        idx = 0
        self.reuse_id_idx = []

        prev_id_idx = None
        cur_group_len = 1
        while True:
            if idx >= len(super_id_idx):
                break
            start_id_idx, start_frame_idx = super_id_idx[idx]
            # Edge case: find where the video ends
            segment_length = 1
            for i in range(1, num_frames):
                end_idx = idx + i
                if end_idx >= len(super_id_idx) - 1:
                    break
                end_id_idx, _ = super_id_idx[end_idx]
                if start_id_idx != end_id_idx:
                    break
                segment_length += 1

            if prev_id_idx != start_id_idx or (refresh_interval > 0 and cur_group_len > refresh_interval):
                # New video
                self.reuse_id_idx.append((start_id_idx, start_frame_idx, 1, True))
                idx += 1
                cur_group_len = 1
                prev_id_idx = start_id_idx
                # self.reuse_id_idx.append((idx, 4, True)) # NOTE: Just for debugging purpose
            else:
                self.reuse_id_idx.append((start_id_idx, start_frame_idx, segment_length, False))
                idx += segment_length
                cur_group_len += segment_length

        if not is_sequential:
            self.pattern = {
                4: [3, 1, 0, 2],
                3: [2, 1, 0, None],
                2: [1, 0, None, None],
                1: [0, None, None, None],
            }
            ref_type = [0, 1, 2, 2]

        else:
            self.pattern = {
                4: [0, 1, 2, 3],
                3: [0, 1, 2, None],
                2: [0, 1, None, None],
                1: [0, None, None, None],
            }
            ref_type = [1, 1, 1, 1]

        self.ref_type = torch.nn.functional.one_hot(torch.tensor(ref_type), num_classes=3).float()

    def __len__(self):
        return len(self.reuse_id_idx)

    def __getitem__(self, idx):
        start_id_idx, super_idx, segment_length, is_new_video = self.reuse_id_idx[idx]
        pattern = self.pattern[segment_length]

        stack = []
        for pat in pattern:
            if pat is not None:
                video_id, *items = super().__getitem__(super_idx + pat)
                stack.append(items)
            if pat is None:
                items = stack[0]
                stack.append(copy.deepcopy(items))
                
        stacked_items = [torch.stack(r, dim=0) for r in zip(*stack)]
        '''
        c = stacked_items[-1]
        # 4, 3, T, W, H
        _, _, T, W, H = c.shape
        c = torch.cat(
            [
                c,
                self.ref_type.view(4, 3, 1, 1, 1).expand(-1, -1, T, W, H)
            ],
            dim=1
        )
        stacked_items[-1] = c
        '''

        return (video_id, is_new_video) + tuple(stacked_items)

    def return_range_per_rank(self, rank, world_size):
        '''Note that indices indicate [start, end) as in Python slicing'''

        start_end_list = []

        start = 0
        end = 0
        # Calculate start and end of each videos
        for idx, (_, _, _, is_new_video) in enumerate(self.reuse_id_idx):
            if is_new_video:
                if idx != 0:
                    start_end_list.append((start, idx))
                start = idx

        start_end_list.append((start, idx + 1))

        print(f"Total number of videos: {len(start_end_list)}")

        cur_rank_start = len(start_end_list) * rank // world_size
        cur_rank_end = len(start_end_list) * (rank + 1) // world_size - 1

        if rank == world_size - 1:
            cur_rank_end = len(start_end_list) - 1

        cur_rank_start = max(cur_rank_start, 0)
        cur_rank_end = min(cur_rank_end, len(start_end_list) - 1)

        start_idx = start_end_list[cur_rank_start][0]
        end_idx = start_end_list[cur_rank_end][1]

        print(f"Rank {rank} will process from {start_idx} to {end_idx}")

        return start_idx, end_idx

    def get_video_range(self, video_id):
        start_idx, end_idx = None, None
        found = False
        for idx, (start_id_idx, *_) in enumerate(self.reuse_id_idx):
            start_id = start_id_idx[0]
            if found:
                if start_id != video_id:
                    end_idx = idx - 1
                    break
            elif start_id == video_id:
                start_idx = idx
                found = True

        assert start_idx is not None, f'Video {video_id} not found'
        if end_idx is None:
            end_idx = len(self.super_id_idx) - 1

        return start_idx, end_idx






def create_test_dataset(base_dataset_class):
    """Factory function to create a train dataset class from any base dataset class."""
    class DynamicTestDataset(ReuseExtractionDatasetMixin, ExtractionDatasetMixin, base_dataset_class):
        def __init__(
                self,
                split,
                base_model_name,
                fps,
                refresh_interval=0,
                is_sequential=False,
                return_pixel_values=False,
                return_input_values=True,
                return_hidden_states=False,
                return_output_states=False,
                return_compressed=False,
                reuse_dataset_info=True,
                use_start_end=False,
                dir_key='feature',
        ):
            # Initialize the base dataset
            base_dataset_class.__init__(
                self, 
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
                dir_key=dir_key
            )

            ExtractionDatasetMixin.__init__(self)

            ReuseExtractionDatasetMixin.__init__(self, num_frames=4, refresh_interval=refresh_interval, is_sequential=is_sequential)
            
    return DynamicTestDataset

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
def main(cfg):
    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)  

    from ...utils.paths import get_path_manager
    path_manager = get_path_manager(cfg.paths)
    from .how2qa import How2qaDataset

    dataset = create_test_dataset(How2qaDataset)(
        pattern=[0, 4, 8, 12, 10, 11],
        split='test',
        base_model_name='openai/clip-vit-large-patch14',
        fps=2,
        return_compressed=True,
        dir_key='feature',
    )

    print(len(dataset))

    for idx, t in enumerate(dataset[0]):
        print(f'Input {idx}: {t.shape}')

    for i in range(200):
        print(dataset[i][0])



if __name__ == '__main__':
    main()

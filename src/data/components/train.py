import hydra
import torch
import numpy as np
import warnings

AUGMENT_NONE_TYPE = 0
AUGMENT_FAR_TYPE = 1
AUGMENT_SAME_TYPE = 2
AUGMENT_SHORT_TYPE = 3
AUGMENT_SHORT_FAR_TYPE = 4

class StackFrameDatasetMixin:
    def __init__(
            self,
            pattern,
            augment_short_ratio=None,
            augment_far_ratio=None,
            augment_same_ratio=None,
            augment_short_far_ratio=None,
            step=None,
        ):
        # ex) [0, 2, 4, 6, 5] would return
        # 1. frame 0, 2, 4, 6, 5
        # 2. frame step, step+2, step+4, step+6, step+5
        self.pattern = pattern
        self.num_frames = len(pattern)
        if step is None:
            step = max(pattern)
        self.step = step
        self.len = 0

        self.stack_frame_states = []

        # Count lenght of videos
        id_start2len = {}
        id_start2start = {}

        prev_video_id = None
        prev_frame_idx = -1
        for super_idx, (video_id, frame_idx) in enumerate(self.dataset_info):
            # Detected new video
            if prev_video_id != video_id or frame_idx != prev_frame_idx + 1:
                if prev_video_id is not None and video_len != 0:
                    id_start2len[(prev_video_id, cur_start_frame_idx)] = video_len
                video_len = 0
                cur_start_frame_idx = frame_idx
                id_start2start[(video_id, cur_start_frame_idx)] = super_idx

            video_len += 1 
            prev_video_id = video_id
            prev_frame_idx = frame_idx

        # AUGMENT_NONE_TYPE
        self.ref_one_hots = {}
        self.ref_masks = {}

        if pattern == [0, 4, 2, 1, 3]:
            self.ref_one_hots[AUGMENT_NONE_TYPE] = torch.nn.functional.one_hot(
                    torch.tensor([0, 0, 1, 2, 2], dtype=torch.long), num_classes=3
            )
            self.ref_masks[AUGMENT_NONE_TYPE] = torch.tensor([
                    [False, False, False, False], # 0: None
                    [True, False, False, False],  # 4: 0
                    [True, True, False, False],   # 2: 0, 4
                    [True, False, True, False],   # 1: 0, 2
                    [False, True, True, False]    # 3: 4, 2
            ])
        elif pattern == [0, 4, 8, 6, 5]:
            self.ref_one_hots[AUGMENT_NONE_TYPE] = torch.nn.functional.one_hot(
                torch.tensor([0, 0, 0, 1, 2], dtype=torch.long), num_classes=3
            )
            self.ref_masks[AUGMENT_NONE_TYPE] = torch.tensor([
                [False, False, False, False], # 0: None
                [ True, False, False, False], # 4: 0
                [False,  True, False, False], # 8: 4
                [False,  True,  True, False], # 6: 4, 8
                [False,  True, False,  True], # 5: 4, 6
            ])
        elif pattern == [0, 4, 8, 6, 7]:
            self.ref_one_hots[AUGMENT_NONE_TYPE] = torch.nn.functional.one_hot(
                torch.tensor([0, 0, 0, 1, 2], dtype=torch.long), num_classes=3
            )
            self.ref_masks[AUGMENT_NONE_TYPE] = torch.tensor([
                [False, False, False, False], # 0: None
                [ True, False, False, False], # 4: 0
                [False,  True, False, False], # 8: 4
                [False,  True,  True, False], # 6: 4, 8
                [False, False,  True,  True], # 7: 6, 8
            ])
        elif pattern == [0, 4, 8, 12, 10, 11]:
            assert augment_far_ratio is None, 'Augmentation is not supported for 6 frame pattern' 
            assert augment_short_ratio is None, 'Augmentation is not supported for 6 frame pattern'
            assert augment_same_ratio is None, 'Augmentation is not supported for 6 frame pattern'
            assert augment_short_far_ratio is None, 'Augmentation is not supported for 6 frame pattern'

            self.ref_one_hots[AUGMENT_NONE_TYPE] = torch.nn.functional.one_hot(
                torch.tensor([0, 0, 0, 0, 1, 2], dtype=torch.long), num_classes=3
            )
            self.ref_masks[AUGMENT_NONE_TYPE] = torch.tensor([
                [False, False, False, False, False], #  0:  None  -> [0]
                [ True, False, False, False, False], #  4:  0     -> [0, 4]
                [False,  True, False, False, False], #  8:  4     -> [0, 4, 8]
                [False, False,  True, False, False], # 12:  8     -> [0, 4, 8, 12]
                [False, False,  True,  True, False], # 10:  8, 12 -> [0, 4, 8, 12, 10]
                [False, False, False,  True,  True], # 11: 12, 10
            ])
        elif pattern == [0, 1, 2, 3, 4]:
            assert augment_short_ratio is None, 'Augmentation is not supported for sequential pattern'
            assert augment_far_ratio is None, 'Augmentation is not supported for sequential pattern'
            assert augment_same_ratio is None, 'Augmentation is not supported for sequential pattern'
            assert augment_short_far_ratio is None, 'Augmentation is not supported for sequential pattern'

            self.ref_one_hots[AUGMENT_NONE_TYPE] = torch.nn.functional.one_hot(
                torch.tensor([0, 0, 0, 0, 0], dtype=torch.long), num_classes=3
            )
            self.ref_masks[AUGMENT_NONE_TYPE] = torch.tensor([
                [False, False, False, False], # 0: None
                [ True, False, False, False], # 1: 0
                [False,  True, False, False], # 2: 1
                [False, False,  True, False], # 3: 2
                [False, False, False,  True], # 4: 3
            ])
        else:
            raise NotImplementedError(f'Unsupported pattern: {pattern}')

        pattern = torch.tensor(pattern, dtype=torch.long)
        max_pattern = max(pattern)
        for (video_id, start_frame_idx), video_len in id_start2len.items():
            video_start_idx = id_start2start[(video_id, start_frame_idx)]
            for start_idx in range(0, video_len, self.step):
                if start_idx + max_pattern >= video_len:
                    break
                self.stack_frame_states.append((
                    AUGMENT_NONE_TYPE,
                    pattern + video_start_idx + start_idx
                ))
        
        len_augment_none = len(self.stack_frame_states)

        if augment_short_ratio is not None:
            assert pattern != [0, 4, 2, 1, 3], 'frame_stack_pattern is using short pattern by default'

            augment_short_samples = []
            self.ref_one_hots[AUGMENT_SHORT_TYPE] = torch.nn.functional.one_hot(
                    torch.tensor([0, 0, 1, 2, 2], dtype=torch.long), num_classes=3
            )
            self.ref_masks[AUGMENT_SHORT_TYPE] = torch.tensor([
                    [False, False, False, False], # 0: None
                    [True, False, False, False],  # 4: 0
                    [True, True, False, False],   # 2: 0, 4
                    [True, False, True, False],   # 1: 0, 2
                    [False, True, True, False]    # 3: 4, 2
            ])
            pattern = torch.tensor(pattern, dtype=torch.long)
            max_pattern = max(pattern)
            for (video_id, start_frame_idx), video_len in id_start2len.items():
                video_start_idx = id_start2start[(video_id, start_frame_idx)]
                for start_idx in range(0, video_len, self.step):
                    if start_idx + max_pattern >= video_len:
                        break
                    augment_short_samples.append((
                        AUGMENT_SHORT_TYPE,
                        pattern + video_start_idx + start_idx
                    ))
            num_target_samples = int(len_augment_none * augment_short_ratio)
            if len(augment_short_samples) < num_target_samples:
                print(f'Unable to populate enough samples for augment_short_type, {len(augment_short_samples)} < {num_target_samples}')
            if len(augment_short_samples) > num_target_samples:
                sample_indices = np.linspace(0, len(augment_short_samples) - 1, num_target_samples, dtype=np.int32)
            else:
                sample_indices = range(len(augment_short_samples))
            
            for sample_idx in sample_indices:
                self.stack_frame_states.append(augment_short_samples[sample_idx])

        if augment_far_ratio is not None:
            # AUGMENT_FAR_TYPE
            augment_far_samples = []
            self.ref_one_hots[AUGMENT_FAR_TYPE] = torch.nn.functional.one_hot(
                torch.tensor([0, 0, 0, 0, 0], dtype=torch.long), num_classes=3
            )
            self.ref_masks[AUGMENT_FAR_TYPE] = torch.tensor([
                [False, False, False, False], #  0: None
                [True, False, False, False],  #  4: 0
                [False, True, False, False],  #  8: 4
                [False, False, True, False],  # 12: 8
                [False, False, False, True]   # 16: 12
            ])
            pattern = torch.tensor([0, 4, 8, 12, 16], dtype=torch.long)
            max_pattern = max(pattern)
            for (video_id, start_frame_idx), video_len in id_start2len.items():
                video_start_idx = id_start2start[(video_id, start_frame_idx)]
                for start_idx in range(0, video_len, max_pattern):
                    if start_idx + max_pattern >= video_len:
                        break
                    augment_far_samples.append((
                        AUGMENT_FAR_TYPE,
                        pattern + video_start_idx + start_idx
                    ))

            num_target_samples = int(len_augment_none * augment_far_ratio)
            if len(augment_far_samples) < num_target_samples:
                print(f'Unable to populate enough samples for augment_far_type, {len(augment_far_samples)} < {num_target_samples}')
            if len(augment_far_samples) > num_target_samples:
                sample_indices = np.linspace(0, len(augment_far_samples) - 1, num_target_samples, dtype=np.int32)
            else:
                sample_indices = range(len(augment_far_samples))
            
            for sample_idx in sample_indices:
                self.stack_frame_states.append(augment_far_samples[sample_idx])

        if augment_same_ratio is not None:
            # AUGMENT_SAME_TYPE
            augment_same_samples = []
            self.ref_one_hots[AUGMENT_SAME_TYPE] = torch.nn.functional.one_hot(
                torch.tensor([0, 2, 1, 0, 0], dtype=torch.long), num_classes=3
            )
            self.ref_masks[AUGMENT_SAME_TYPE] = torch.tensor([
                [False, False, False, False], # 0: None
                [ True, False, False, False], # 0: 0
                [False,  True, False, False], # 0: 0
                [False, False,  True, False], # 0: 0
                [False, False, False,  True]  # 0: 0
            ])
            pattern = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
            num_target_samples = int(len_augment_none * augment_same_ratio)
            sample_indices = np.linspace(0, len_augment_none - 1, num_target_samples, dtype=np.int32)
            for sample_idx in sample_indices:
                _, pattern = self.stack_frame_states[sample_idx]
                frame_idx = pattern[0]
                pattern = torch.tensor([frame_idx] * len(pattern), dtype=torch.long)
                self.stack_frame_states.append((AUGMENT_SAME_TYPE, pattern))

        if augment_short_far_ratio is not None:
            # AUGMENT_FAR_TYPE
            augment_short_far_samples = []
            self.ref_one_hots[AUGMENT_SHORT_FAR_TYPE] = torch.nn.functional.one_hot(
                torch.tensor([0, 0, 0, 0, 0], dtype=torch.long), num_classes=3
            )
            self.ref_masks[AUGMENT_SHORT_FAR_TYPE] = torch.tensor([
                [False, False, False, False], # None
                [True, False, False, False],  # 0
                [True, False, False, False],  # 0
                [False, False, True, False],  # 4
                [False, False, False, True]   # 8
            ])
            pattern = torch.tensor([0, 4, 4, 8, 12], dtype=torch.long)
            max_pattern = max(pattern)
            for (video_id, frame_idx), video_len in id_start2len.items():
                video_start_idx = id_start2start[(video_id, frame_idx)]
                for start_idx in range(0, video_len, max_pattern):
                    if start_idx + max_pattern >= video_len:
                        break
                    augment_short_far_samples.append((
                        AUGMENT_SHORT_FAR_TYPE,
                        pattern + video_start_idx + start_idx
                    ))

            num_target_samples = int(len_augment_none * augment_short_far_ratio)
            if len(augment_short_far_samples) < num_target_samples:
                print(f'Unable to populate enough samples for augment_short_far_type, {len(augment_short_far_samples)} < {num_target_samples}')
            if len(augment_short_far_samples) > num_target_samples:
                sample_indices = np.linspace(0, len(augment_short_far_samples) - 1, num_target_samples, dtype=np.int32)
            else:
                sample_indices = range(len(augment_short_far_samples))
            
            for sample_idx in sample_indices:
                self.stack_frame_states.append(augment_short_far_samples[sample_idx])


    def __getitem__(self, idx):
        augment_type, pattern = self.stack_frame_states[idx]

        rets = []
        for p in pattern:
            rets.append(super().__getitem__(p))

        if len(rets) == 1:
            rets = rets[0]
        else:
            rets = list(zip(*rets))
            rets = [torch.stack(r, dim=0) for r in rets]

        ref_one_hot = self.ref_one_hots[augment_type]
        rets.append(ref_one_hot)

        ref_mask = self.ref_masks[augment_type]
        rets.append(ref_mask)

        return rets

    def __len__(self):
        return len(self.stack_frame_states)

def create_train_dataset(base_dataset_class):
    """Factory function to create a train dataset class from any base dataset class."""
    class DynamicTrainDataset(StackFrameDatasetMixin, base_dataset_class):
        def __init__(
                self,
                pattern,
                split,
                base_model_name,
                fps,
                return_pixel_values=False,
                return_input_values=True,
                return_hidden_states=False,
                return_output_states=False,
                return_compressed=False,
                reuse_dataset_info=True,
                augment_short_ratio=None,
                augment_far_ratio=None,
                augment_same_ratio=None,
                augment_short_far_ratio=None,
                step=None,
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
            
            # Initialize the mixin
            StackFrameDatasetMixin.__init__(
                self,
                pattern,
                augment_short_ratio=augment_short_ratio,
                augment_far_ratio=augment_far_ratio,
                augment_same_ratio=augment_same_ratio,
                augment_short_far_ratio=augment_short_far_ratio,
                step=step,
            )
    
    return DynamicTrainDataset

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
def main(cfg):
    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)  

    from ...utils.paths import get_path_manager
    path_manager = get_path_manager(cfg.paths)
    from .how2qa import How2qaDataset

    dataset = create_train_dataset(How2qaDataset)(
        pattern=[0, 4, 8, 12, 10, 11],
        split='test',
        base_model_name='openai/clip-vit-large-patch14',
        fps=2,
        return_compressed=True,
        # augment_short_ratio=1.,
        # augment_far_ratio=1.,
        # augment_same_ratio=1.,
        # augment_short_far_ratio=1.,
        dir_key='feature',
    )

    print(len(dataset))

    for idx, t in enumerate(dataset[0]):
        print(f'Input {idx}: {t.shape}')

    for i in range(200):
        print(dataset[i][0])



if __name__ == '__main__':
    main()

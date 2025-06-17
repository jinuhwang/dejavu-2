#!/usr/bin/env python
import torch
from pathlib import Path
import numpy as np
import warnings

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from functools import partial

# Filter and ignore the specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable.*")

def get_cropped_video_path(video_path, start, end):
    video_path = Path(video_path)
    return video_path.parent / f'{video_path.stem}_{start}_{end}.mp4'

def filter_existing_files(output_paths, *others):
    output_paths_len = len(output_paths)
    assert all(len(o) == output_paths_len for o in others), "All lists must have the same length"

    output_paths_filtered = []
    others_filtered = [[] for _ in range(len(others))]

    num_skipped = 0
    for i in range(output_paths_len):
        output_path = Path(output_paths[i])
        if not output_path.exists():
            output_paths_filtered.append(output_path)
            for j, o in enumerate(others):
                others_filtered[j].append(o[i])
        else:
            num_skipped += 1

    print(f"Skipping {num_skipped} existing files out of {output_paths_len}")

    ret = (output_paths_filtered, *others_filtered)

    return ret


def split_per_rank(rank, world_size, *items):
    assert world_size > 0, "World size must be positive"
    assert rank >= 0 and rank < world_size, f"Rank must be in [0, {world_size})"
    assert len(items) > 0, "At least one item must be provided"
    assert all(len(items[0]) == len(i) for i in items), "All items must have the same length"

    if len(items[0]) < world_size:
        return items

    ret = [[] for _ in range(len(items))]

    items_per_rank = len(items[0]) // world_size

    start_idx = rank * items_per_rank
    if rank == world_size - 1:
        end_idx = len(items[0])
    else:
        end_idx = (rank + 1) * items_per_rank

    for i in range(len(items)):
        ret[i] = items[i][start_idx:end_idx]

    return ret

def count_sampled_frames(video_path, sample_fps=1):
    import cv2
    """
    Count the number of frames that would be sampled from a video.

    Args:
        video_path (str): Path to the video file.
        sample_fps (int): The number of frames to sample per second.

    Returns:
        int: Number of frames that would be sampled.
    """
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    # Get video FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # If FPS is not available, count frames manually
    if not video_fps:
        total_frames = 0
        while True:
            # Try to read a frame
            has_frame, _ = cap.read()
            if has_frame:
                total_frames += 1
            else:
                break
    else:
        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Close the video file
    cap.release()

    # Calculate video duration in seconds
    video_duration = total_frames / video_fps if video_fps else total_frames
    # Calculate number of frames to be sampled
    sampled_frames = int(video_duration * sample_fps)

    return sampled_frames


def sample_frames(video_path, sampling_fps=1):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'Failed to open video: {video_path}')
        return []

    # Get the frame rate
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize a counter for the time in seconds
    time_seconds = 0

    frames = []
    while True:
        # Calculate the frame number corresponding to the current time
        frame_number = round(time_seconds * frame_rate)

        # Break the loop if the frame number exceeds the total number of frames
        if frame_number >= total_frames:
            break

        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Frame read successfully
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        # Increment the time counter by wanted sampling interval
        time_seconds += (1 / sampling_fps)

    # Release the video capture object
    cap.release()
    print(len(frames))
    return frames

def fill_list_to_length(a, num_frames):
    """
    Fill the list a to length 'frames' by repeating its elements.
    The elements from the end of the list are repeated more if necessary.
    """
    filled_list = []
    len_a = len(a)
    repeats = num_frames // len_a  # Number of times the whole list can be repeated
    additional_elements = num_frames % len_a  # Number of additional elements needed

    for i in range(len_a):
        # Repeat each element 'repeats' times or 'repeats + 1' times for the last few elements
        repeat_times = repeats + 1 if i >= len_a - additional_elements else repeats
        filled_list.extend([a[i]] * repeat_times)

    return filled_list


def make_best_indices(vid_len, sampling_fps=3, video_fps=30):
    indices = np.arange(0, vid_len, video_fps / sampling_fps, dtype=int)
    return indices


def sample_frames_with_gap(video_path, sampling_fps=3):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    # Get the frame rate
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = make_best_indices(total_frames, sampling_fps=sampling_fps, video_fps=video_fps)
    frames = []
    for frame_idx in frame_indices:
        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Frame read successfully
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames


def sample_frames_all(video_path, frame_rate=1):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    while True:
        # Read the frame
        ret, frame = cap.read()

        if not ret:
            break

        # Frame read successfully
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Release the video capture object
    cap.release()
    print(len(frames))
    return frames

def sample_frames_CLIP4Clip(video_path, frame_rate=1):
    import cv2
    
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Samples a frame sample_fp X frames.
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    total_duration = (frameCount + fps - 1) // fps
    start_sec, end_sec = 0, total_duration

    interval = 1
    if frame_rate > 0:
        interval = fps // frame_rate
    else:
        frame_rate = fps
    if interval == 0: interval = 1

    inds = [ind for ind in np.arange(0, fps, interval)]
    assert len(inds) >= frame_rate
    inds = inds[:frame_rate]

    ret = True
    frames, included = [], []

    for sec in np.arange(start_sec, end_sec + 1):
        if not ret: break
        sec_base = int(sec * fps)
        for ind in inds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

    cap.release()
    print(len(frames))
    return frames

def load_npy_embedding(feature_path, return_pt=True):
    """
    Load a feature vector from a .npy file.

    Args:
        feature_path (str): Path to the .npy file.
        return_pt (bool): If True, returns a PyTorch tensor. Otherwise, returns a NumPy array.

    Returns:
        embedding (np.ndarray or torch.Tensor): The feature vector.
    """
    # Load the .npy file as a NumPy array
    embedding = np.load(feature_path)

    # Convert to PyTorch tensor if needed
    if return_pt:
        embedding = torch.from_numpy(embedding)

    return embedding

def get_all_frames(video_path):
    import cv2

    frames = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'Failed to open video: {video_path}')
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame read successfully
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames

def fill_list_to_length(a, num_frames):
    """
    Fill the list a to length 'frames' by repeating its elements.
    The elements from the end of the list are repeated more if necessary.
    """
    filled_list = []
    len_a = len(a)
    repeats = num_frames // len_a  # Number of times the whole list can be repeated
    additional_elements = num_frames % len_a  # Number of additional elements needed

    for i in range(len_a):
        # Repeat each element 'repeats' times or 'repeats + 1' times for the last few elements
        repeat_times = repeats + 1 if i >= len_a - additional_elements else repeats
        filled_list.extend([a[i]] * repeat_times)

    return filled_list


def make_best_indices(vid_len, sampling_fps=3, video_fps=30):
    indices = np.arange(0, vid_len, video_fps / sampling_fps, dtype=int)
    return indices


def sample_frames_with_gap(video_path, sampling_fps=3):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    # Get the frame rate
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = make_best_indices(total_frames, sampling_fps=sampling_fps, video_fps=video_fps)
    frames = []
    for frame_idx in frame_indices:
        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Frame read successfully
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames


def sample_frames_all(video_path, frame_rate=1):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    while True:
        # Read the frame
        ret, frame = cap.read()

        if not ret:
            break

        # Frame read successfully
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames

def sample_frames_CLIP4Clip(video_path, frame_rate=1):
    import cv2
    
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Samples a frame sample_fp X frames.
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    total_duration = (frameCount + fps - 1) // fps
    start_sec, end_sec = 0, total_duration

    interval = 1
    if frame_rate > 0:
        interval = fps // frame_rate
    else:
        frame_rate = fps
    if interval == 0: interval = 1

    inds = [ind for ind in np.arange(0, fps, interval)]
    assert len(inds) >= frame_rate
    inds = inds[:frame_rate]

    ret = True
    frames, included = [], []

    for sec in np.arange(start_sec, end_sec + 1):
        if not ret: break
        sec_base = int(sec * fps)
        for ind in inds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

    cap.release()
    print(len(frames))
    return frames

def save_embedding(embedding, feature_path, mkdir=True):
    if mkdir:
        feature_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.float().cpu().numpy()
    np.savez_compressed(feature_path, embeddings=embedding)

def load_embedding(feature_path, return_pt=True):
    """
    Load a feature vector from a .npz file.

    Args:
        feature_path (str): Path to the .npz file.

    Returns:
        embedding (np.ndarray): The feature vector.
    """
    with np.load(feature_path, allow_pickle=True) as data:
        embedding = data['embeddings']
    if return_pt:
        embedding = torch.from_numpy(embedding)
    return embedding

def load_npy_embedding(feature_path, return_pt=True):
    """
    Load a feature vector from a .npy file.

    Args:
        feature_path (str): Path to the .npy file.
        return_pt (bool): If True, returns a PyTorch tensor. Otherwise, returns a NumPy array.

    Returns:
        embedding (np.ndarray or torch.Tensor): The feature vector.
    """
    # Load the .npy file as a NumPy array
    embedding = np.load(feature_path)

    # Convert to PyTorch tensor if needed
    if return_pt:
        embedding = torch.from_numpy(embedding)

    return embedding


def ray_get_with_tqdm(ret, num_results=None):
    import ray
    from tqdm import tqdm

    if num_results is None:
        num_results = len(ret)

    with tqdm(total=num_results) as pbar:
        while ret:
            # Get results as they complete
            done_refs, ret = ray.wait(ret)
            for ref in done_refs:
                pbar.update(1)
                ray.get(ref)

def count_available_features(
        dataset,
        fps,
        split,
        base_model_name,
        video_ids,
        starts=None,
        ends=None,
        check_compressed=True,
        use_feature_path_v2=False,
        dir_key='feature',
    ):
    '''
    returns list in the format of (youtube_id, start, end, num_frames)
    '''
    from .paths import get_path_manager
    path_manager = get_path_manager()

    ret = []

    get_feature_path_fn = partial(
        path_manager.get_feature_path,
        dataset=dataset,
        base_model_name=base_model_name,
        fps=fps,
        split=split,
        dir_key=dir_key,
    )

    for video_idx, video_id in enumerate(video_ids):
        if starts is not None:
            start = starts[video_idx]
            end = ends[video_idx]
        else:
            start = 0
            end = None

        frame_cnt = 0
        while end is None or start + frame_cnt <= end:
            pixel_path = get_feature_path_fn(video_id, 'i', start + frame_cnt, use_v2=use_feature_path_v2)
            exists = pixel_path.exists()
            if check_compressed:
                compressed_path = get_feature_path_fn(video_id, 'c', start + frame_cnt, use_v2=use_feature_path_v2)
                exists = exists and compressed_path.exists()
            if not exists:
                break
            frame_cnt += 1
        end = start + frame_cnt
        ret.append((video_id, start, end, frame_cnt))

    return ret

def get_resolution(base_model_name):
    if base_model_name == 'openai/clip-vit-large-patch14':
        resolution = 256
    elif base_model_name == 'openai/clip-vit-base-patch16':
        resolution = 224
    elif base_model_name == 'google/siglip-base-patch16-224':
        resolution = 224
    else:
        raise NotImplementedError(f'Base model name {base_model_name} is not supported')

    return resolution
# Copyright 2025 VideoPrism Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for video loading and preprocessing."""

import numpy as np


def load_video(
    video_path: str,
    num_frames: int = 16,
    target_size: int = 288,
    resize_mode: str = "center_crop",
) -> np.ndarray:
  """Loads a video and preprocesses it for VideoPrism models.

  Args:
    video_path: Path to the video file.
    num_frames: Number of frames to sample from the video. Default is 16 for
      base models.
    target_size: Target height and width (square). Default is 288.
    resize_mode: How to resize frames. Options:
      - "center_crop": Resize shortest side to target_size, then center crop.
      - "resize": Simple resize to target_size x target_size (may distort).

  Returns:
    A numpy array of shape [num_frames, target_size, target_size, 3] with RGB
    values normalized to [0.0, 1.0].

  Raises:
    ImportError: If cv2 (OpenCV) is not installed.
  """
  try:
    import cv2
  except ImportError as e:
    raise ImportError(
        "OpenCV is required for video loading. "
        "Install it with: pip install opencv-python"
    ) from e

  # Open the video file
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise ValueError(f"Could not open video file: {video_path}")

  # Get total number of frames
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames < num_frames:
    raise ValueError(
        f"Video has only {total_frames} frames, but {num_frames} requested"
    )

  # Sample frame indices uniformly
  frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

  frames = []
  for frame_idx in frame_indices:
    # Seek to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
      raise ValueError(f"Could not read frame {frame_idx} from {video_path}")

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame
    if resize_mode == "center_crop":
      frame = _center_crop_resize(frame, target_size)
    elif resize_mode == "resize":
      frame = cv2.resize(frame, (target_size, target_size))
    else:
      raise ValueError(f"Unknown resize_mode: {resize_mode}")

    frames.append(frame)

  cap.release()

  # Stack frames and normalize to [0.0, 1.0]
  video_array = np.stack(frames, axis=0).astype(np.float32) / 255.0

  return video_array


def _center_crop_resize(frame: np.ndarray, target_size: int) -> np.ndarray:
  """Resizes the shortest side to target_size, then center crops.

  Args:
    frame: Input frame of shape [height, width, channels].
    target_size: Target size for both height and width.

  Returns:
    Resized and cropped frame of shape [target_size, target_size, channels].
  """
  import cv2

  h, w = frame.shape[:2]

  # Resize shortest side to target_size
  if h < w:
    new_h = target_size
    new_w = int(w * (target_size / h))
  else:
    new_w = target_size
    new_h = int(h * (target_size / w))

  frame = cv2.resize(frame, (new_w, new_h))

  # Center crop
  h, w = frame.shape[:2]
  start_y = (h - target_size) // 2
  start_x = (w - target_size) // 2
  cropped = frame[start_y : start_y + target_size, start_x : start_x + target_size]

  return cropped


def load_video_batch(
    video_paths: list[str],
    num_frames: int = 16,
    target_size: int = 288,
    resize_mode: str = "center_crop",
) -> np.ndarray:
  """Loads multiple videos as a batch.

  Args:
    video_paths: List of paths to video files.
    num_frames: Number of frames to sample from each video.
    target_size: Target height and width (square).
    resize_mode: How to resize frames.

  Returns:
    A numpy array of shape [batch_size, num_frames, target_size, target_size, 3]
    with RGB values normalized to [0.0, 1.0].
  """
  videos = [
      load_video(path, num_frames, target_size, resize_mode)
      for path in video_paths
  ]
  return np.stack(videos, axis=0)

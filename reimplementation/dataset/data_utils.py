"""
Data utilities to replace mmcv file I/O functions.
Provides dump, load, and file checking utilities without mmcv dependencies.
"""

import copy
import os
import pickle
import json
import warnings
from pathlib import Path
from typing import Any, Optional, Union
import torch
from tqdm import tqdm
import numpy as np
from collections.abc import Sequence
from ..models.common.box3d import *
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def is_filepath(path: Union[str, Path]) -> bool:
    """Check if a path exists.

    Args:
        path: File path to check

    Returns:
        True if file exists, False otherwise
    """
    return os.path.isfile(path)


def check_file_exist(filename: Union[str, Path], msg: Optional[str] = None):
    """Check if a file exists, raise error if not.

    Args:
        filename: File path to check
        msg: Optional error message

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.isfile(filename):
        if msg is None:
            msg = f'File "{filename}" does not exist'
        raise FileNotFoundError(msg)


def is_str(x) -> bool:
    """Check if input is a string instance.

    Args:
        x: Input to check

    Returns:
        True if x is a string, False otherwise
    """
    return isinstance(x, str)


def dump(obj: Any,
         file: Optional[Union[str, Path]] = None,
         file_format: Optional[str] = None,
         **kwargs):
    """Dump data to pickle/json file or string.

    This is a simplified replacement for mmcv.dump that handles
    pickle and json formats for local files only.

    Args:
        obj: Python object to dump
        file: File path or file object. If None, returns string
        file_format: File format ('pkl', 'pickle', 'json').
                     Auto-detected from file extension if not specified
        **kwargs: Additional arguments passed to pickle.dump or json.dump

    Returns:
        None if file is specified, otherwise returns dumped string

    Examples:
        >>> dump({'a': 1}, 'data.pkl')  # Save to pickle file
        >>> dump({'a': 1}, 'data.json')  # Save to JSON file
        >>> s = dump({'a': 1})  # Dump to string (pickle format)
    """
    if isinstance(file, Path):
        file = str(file)

    # Determine file format
    if file_format is None:
        if is_str(file):
            file_format = file.split('.')[-1]
        elif file is None:
            file_format = 'pkl'  # Default to pickle
        else:
            raise ValueError('file_format must be specified')

    # Normalize format
    if file_format in ['pkl', 'pickle']:
        handler = PickleHandler()
    elif file_format == 'json':
        handler = JsonHandler()
    else:
        raise ValueError(f'Unsupported file format: {file_format}')

    # Dump to file or string
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        # Use write mode: 'wb' for pickle, 'w' for json
        write_mode = 'wb' if file_format in ['pkl', 'pickle'] else 'w'
        with open(file, write_mode) as f:
            handler.dump_to_fileobj(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def load(file: Union[str, Path],
         file_format: Optional[str] = None,
         **kwargs) -> Any:
    """Load data from pickle/json file or string.

    This is a simplified replacement for mmcv.load.

    Args:
        file: File path, file object, or string to load from
        file_format: File format ('pkl', 'pickle', 'json')
                     Auto-detected from file extension if not specified
        **kwargs: Additional arguments passed to pickle.load or json.load

    Returns:
        Loaded Python object

    Examples:
        >>> data = load('data.pkl')  # Load from pickle file
        >>> data = load('data.json')  # Load from JSON file
    """
    if isinstance(file, Path):
        file = str(file)

    # Determine file format
    if file_format is None:
        if is_str(file):
            file_format = file.split('.')[-1]
        else:
            raise ValueError('file_format must be specified')

    # Normalize format
    if file_format in ['pkl', 'pickle']:
        handler = PickleHandler()
    elif file_format == 'json':
        handler = JsonHandler()
    else:
        raise ValueError(f'Unsupported file format: {file_format}')

    # Load from file or string
    if is_str(file):
        # Use read mode: 'rb' for pickle, 'r' for json
        read_mode = 'rb' if file_format in ['pkl', 'pickle'] else 'r'
        with open(file, read_mode) as f:
            return handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, 'read'):
        return handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def track_iter_progress(iterable, desc: Optional[str] = None):
    """Track iteration progress with tqdm.

    This is a replacement for mmcv.track_iter_progress.

    Args:
        iterable: Iterable to track
        desc: Description for progress bar

    Yields:
        Items from iterable

    Examples:
        >>> for item in track_iter_progress(items, desc='Processing'):
        ...     process(item)
    """
    return tqdm(iterable, desc=desc)


class PickleHandler:
    """Handler for pickle files."""

    str_like = False
    mode = 'wb'

    def load_from_fileobj(self, file, **kwargs):
        """Load from file object."""
        return pickle.load(file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        """Dump to bytes string."""
        kwargs.setdefault('protocol', 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        """Dump to file object."""
        kwargs.setdefault('protocol', 2)
        pickle.dump(obj, file, **kwargs)


class JsonHandler:
    """Handler for JSON files."""

    str_like = True
    mode = 'w'

    def load_from_fileobj(self, file, **kwargs):
        """Load from file object."""
        return json.load(file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        """Dump to string."""
        kwargs.setdefault('indent', 2)
        return json.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        """Dump to file object."""
        kwargs.setdefault('indent', 2)
        json.dump(obj, file, **kwargs)


def to_tensor(data):
    """Convert Python objects to torch.Tensor without MMCV dependency.
    
    Supported types:
    - torch.Tensor → returned as is
    - numpy.ndarray → converted via torch.from_numpy
    - Sequence (list, tuple) → converted via torch.tensor
    - int → LongTensor
    - float → FloatTensor
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.tensor([data], dtype=torch.long)
    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float)
    else:
        raise TypeError(f"type {type(data)} cannot be converted to torch.Tensor.")

def imnormalize_cv2(img, mean, std, to_rgb=True):
    if not CV2_AVAILABLE:
        raise ImportError('cv2 is required for normalization')
    img = img.astype(np.float32)

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = (img - mean) / std
    return img


def bgr2hsv(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to HSV color space.

    Args:
        img (np.ndarray): BGR image array (in [0, 255] range for both uint8 and float32)

    Returns:
        np.ndarray: HSV image array with H in [0, 360), S in [0, 1], V in [0, 1]
    """
    if not CV2_AVAILABLE:
        raise ImportError('cv2 is required for BGR to HSV conversion')

    # cv2.cvtColor expects uint8 or float32
    # HSV output: H in [0, 180), S in [0, 255], V in [0, 255] for uint8
    # HSV output: H in [0, 360), S in [0, 1], V in [0, 1] for float32 IN [0, 1] RANGE

    if img.dtype == np.uint8:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Convert to mmcv format: H in [0, 360), S in [0, 1], V in [0, 1]
        hsv = hsv.astype(np.float32)
        hsv[..., 0] = hsv[..., 0] * 2  # H: [0, 180) -> [0, 360)
        hsv[..., 1] = hsv[..., 1] / 255.0  # S: [0, 255] -> [0, 1]
        hsv[..., 2] = hsv[..., 2] / 255.0  # V: [0, 255] -> [0, 1]
    elif img.dtype == np.float32:
        # BUG FIX: cv2 expects float32 images in [0, 1] range, but ours are in [0, 255]!
        # We need to scale to [0, 1] first
        img_scaled = img / 255.0
        hsv = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2HSV)
        # cv2 outputs H in [0, 360), S in [0, 1], V in [0, 1] for float32
    else:
        # Convert to float32 first, then scale
        img_float = img.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)

    return hsv


def hsv2bgr(img: np.ndarray) -> np.ndarray:
    """Convert an HSV image to BGR color space.

    Args:
        img (np.ndarray): HSV image array with H in [0, 360), S in [0, 1], V in [0, 1]

    Returns:
        np.ndarray: BGR image array in [0, 255] range as float32
    """
    if not CV2_AVAILABLE:
        raise ImportError('cv2 is required for HSV to BGR conversion')

    # Input is expected to be float32 with H in [0, 360), S in [0, 1], V in [0, 1]
    # cv2 expects H in [0, 360), S in [0, 1], V in [0, 1] for float32

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # BUG FIX: cv2 outputs BGR in [0, 1] range for float32 input
    # Scale back to [0, 255] to match the expected range
    bgr = bgr * 255.0

    return bgr


def imread(img_or_path: Union[np.ndarray, str, Path],
           flag: str = 'color',
           channel_order: str = 'bgr',
           backend: Optional[str] = None) -> np.ndarray:
    """Read an image.

    This is a simplified replacement for mmcv.imread that uses cv2 or PIL.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale`, `unchanged`.
            Default: 'color'
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
            Default: 'bgr'
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `None`. If None, will try cv2 first, then pillow.
            Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> img_path = '/path/to/img.jpg'
        >>> img = imread(img_path)
        >>> img = imread(img_path, flag='color', channel_order='rgb')
        >>> img = imread(img_path, flag='grayscale')
    """
    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        # Determine backend
        if backend is None:
            if CV2_AVAILABLE:
                backend = 'cv2'
            elif PIL_AVAILABLE:
                backend = 'pillow'
            else:
                raise ImportError(
                    'Either cv2 or PIL must be installed to read images')

        # Read image with specified backend
        if backend == 'cv2':
            if not CV2_AVAILABLE:
                raise ImportError('cv2 is not installed')

            # Map flag to cv2 flag
            if flag == 'color':
                cv2_flag = cv2.IMREAD_COLOR
            elif flag == 'grayscale':
                cv2_flag = cv2.IMREAD_GRAYSCALE
            elif flag == 'unchanged':
                cv2_flag = cv2.IMREAD_UNCHANGED
            elif flag == 'color_ignore_orientation':
                cv2_flag = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            elif flag == 'grayscale_ignore_orientation':
                cv2_flag = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION
            else:
                raise ValueError(f'Unsupported flag: {flag}')

            img = cv2.imread(img_or_path, cv2_flag)
            if img is None:
                raise ValueError(f'Failed to read image: {img_or_path}')

            # Convert channel order if needed
            if flag in ['color', 'color_ignore_orientation'] and channel_order == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        elif backend == 'pillow':
            if not PIL_AVAILABLE:
                raise ImportError('PIL is not installed')

            img = Image.open(img_or_path)

            # Convert based on flag
            if flag == 'color':
                img = img.convert('RGB')
            elif flag == 'grayscale':
                img = img.convert('L')
            elif flag == 'unchanged':
                pass  # Keep as is
            elif flag in ['color_ignore_orientation', 'grayscale_ignore_orientation']:
                # PIL doesn't auto-rotate by default, so just convert
                if 'color' in flag:
                    img = img.convert('RGB')
                else:
                    img = img.convert('L')
            else:
                raise ValueError(f'Unsupported flag: {flag}')

            # Convert to numpy array
            img = np.array(img)

            # Convert channel order if needed
            if flag in ['color', 'color_ignore_orientation']:
                if channel_order == 'bgr' and img.ndim == 3:
                    # PIL uses RGB, convert to BGR if requested
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if CV2_AVAILABLE else img[:, :, ::-1]

            return img

        else:
            raise ValueError(f'Unsupported backend: {backend}')
    else:
        raise TypeError('"img_or_path" must be a numpy array or a str or '
                        'a pathlib.Path object')





def box3d_to_corners(box3d):
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners


def plot_rect3d_on_img(
    img, num_rects, rect_corners, color=(0, 255, 0), thickness=1
):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    h, w = img.shape[:2]
    for i in range(num_rects):
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
        for start, end in line_indices:
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)
                or (corners[start, 0] >= w or corners[start, 0] < 0)
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)
                or (corners[end, 0] >= w or corners[end, 0] < 0)
            ):
                continue
            if isinstance(color[0], int):
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color[i],
                    thickness,
                    cv2.LINE_AA,
                )

    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(
    bboxes3d, raw_img, lidar2img_rt, img_metas=None, color=(0, 255, 0), thickness=1
):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    # corners_3d = bboxes3d.corners
    corners_3d = box3d_to_corners(bboxes3d)
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4):
    img = img.copy()
    N = points.shape[0]
    points = points.cpu().numpy()
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = (
        np.sum(points[:, :, None] * lidar2img_rt[:3, :3], axis=-1)
        + lidar2img_rt[:3, 3]
    )
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)
    pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]
    pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)

    for i in range(N):
        for point in pts_2d[i]:
            if isinstance(color[0], int):
                color_tmp = color
            else:
                color_tmp = color[i]
            cv2.circle(img, point.tolist(), circle, color_tmp, thickness=-1)
    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_bev(
    bboxes_3d, bev_size, bev_range=115, color=(255, 0, 0), thickness=3):
    if isinstance(bev_size, (list, tuple)):
        bev_h, bev_w = bev_size
    else:
        bev_h, bev_w = bev_size, bev_size
    bev = np.zeros([bev_h, bev_w, 3])

    marking_color = (127, 127, 127)
    bev_resolution = bev_range / bev_h
    for cir in range(int(bev_range / 2 / 10)):
        cv2.circle(
            bev,
            (int(bev_h / 2), int(bev_w / 2)),
            int((cir + 1) * 10 / bev_resolution),
            marking_color,
            thickness=thickness,
        )
    cv2.line(
        bev,
        (0, int(bev_h / 2)),
        (bev_w, int(bev_h / 2)),
        marking_color,
    )
    cv2.line(
        bev,
        (int(bev_w / 2), 0),
        (int(bev_w / 2), bev_h),
        marking_color,
    )
    if len(bboxes_3d) != 0:
        bev_corners = box3d_to_corners(bboxes_3d)[:, [0, 3, 4, 7]][
            ..., [0, 1]
        ]
        xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
        ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
        for obj_idx, (x, y) in enumerate(zip(xs, ys)):
            for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                if isinstance(color[0], (list, tuple)):
                    tmp = color[obj_idx]
                else:
                    tmp = color
                cv2.line(
                    bev,
                    (int(x[p1]), int(y[p1])),
                    (int(x[p2]), int(y[p2])),
                    tmp,
                    thickness=thickness,
                )
    return bev.astype(np.uint8)


def draw_lidar_bbox3d(bboxes_3d, imgs, lidar2imgs, color=(255, 0, 0)):
    vis_imgs = []
    for i, (img, lidar2img) in enumerate(zip(imgs, lidar2imgs)):
        vis_imgs.append(
            draw_lidar_bbox3d_on_img(bboxes_3d, img, lidar2img, color=color)
        )

    num_imgs = len(vis_imgs)
    if num_imgs < 4 or num_imgs % 2 != 0:
        vis_imgs = np.concatenate(vis_imgs, axis=1)
    else:
        vis_imgs = np.concatenate([
            np.concatenate(vis_imgs[:num_imgs//2], axis=1),
            np.concatenate(vis_imgs[num_imgs//2:], axis=1)
        ], axis=0)

    bev = draw_lidar_bbox3d_on_bev(bboxes_3d, vis_imgs.shape[0], color=color)
    vis_imgs = np.concatenate([bev, vis_imgs], axis=1)
    return vis_imgs


__all__ = [
    'dump',
    'load',
    'is_filepath',
    'check_file_exist',
    'is_str',
    'track_iter_progress',
    'imread',
]
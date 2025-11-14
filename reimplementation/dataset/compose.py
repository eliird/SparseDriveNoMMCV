import collections.abc
from typing import Dict, Any, Optional


# Pipeline registry to replace mmcv's PIPELINES
PIPELINES: Dict[str, type] = {}


def build_from_cfg(cfg: Optional[Dict[str, Any]], registry: Optional[Dict[str, type]] = None):
    """Build a pipeline transform from config dict.

    Args:
        cfg (dict or None): Configuration dict with 'type' key, or None
        registry (dict, optional): Registry dict mapping type names to classes

    Returns:
        object: Built transform instance, or None if cfg is None

    Raises:
        TypeError: If cfg is not a dict
        KeyError: If 'type' is missing or not in registry
    """
    if cfg is None:
        return None

    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

    if 'type' not in cfg:
        raise KeyError('cfg must contain the key "type"')

    # Make a copy to avoid modifying the original config
    cfg = cfg.copy()
    transform_type = cfg.pop('type')

    # Use provided registry or default to PIPELINES
    if registry is None:
        registry = PIPELINES

    if transform_type not in registry:
        raise KeyError(
            f'Unrecognized transform type: {transform_type}. '
            f'Available types: {list(registry.keys())}'
        )

    transform_class = registry[transform_type]
    return transform_class(**cfg)


def register_pipeline(name: str, pipeline_class: type):
    """Register a new pipeline transform.

    Args:
        name (str): Type name for the transform
        pipeline_class (type): The class to register
    """
    PIPELINES[name] = pipeline_class


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


# Auto-register all pipeline transforms from pipelines.py and vectorize.py
def _register_all_pipelines():
    """Automatically register all pipeline transform classes."""
    try:
        from . import pipelines
        from . import vectorize

        # Register all pipeline transform classes from pipelines.py
        pipeline_classes = [
            'LoadMultiViewImageFromFiles',
            'LoadPointsFromFile',
            'ResizeCropFlipImage',
            'MultiScaleDepthMapGenerator',
            'BBoxRotation',
            'PhotoMetricDistortionMultiViewImage',
            'NormalizeMultiviewImage',
            'CircleObjectRangeFilter',
            'InstanceNameFilter',
            'NuScenesSparse4DAdaptor',
            'Collect',
        ]

        for class_name in pipeline_classes:
            if hasattr(pipelines, class_name):
                register_pipeline(class_name, getattr(pipelines, class_name))

        # Register pipeline transform classes from vectorize.py
        vectorize_classes = [
            'VectorizeMap',
        ]

        for class_name in vectorize_classes:
            if hasattr(vectorize, class_name):
                register_pipeline(class_name, getattr(vectorize, class_name))
    except ImportError:
        # pipelines/vectorize module not available yet, will be registered later
        pass


# Register pipelines on module import
_register_all_pipelines()


__all__ = ['Compose', 'PIPELINES', 'build_from_cfg', 'register_pipeline']
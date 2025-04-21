#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base classes for tracks and trackers."""

__all__ = [
    "Detection",
    "Track",
    "Tracker",
]

from abc import ABC, abstractmethod
from timeit import default_timer as timer

import numpy as np

from mon.core import TrackState
from mon.vision import types


# region Track

class Detection:
    """An instance of a track in a frame. This class is mainly used to wrap and
    pass data between detectors and trackers.
    
    Args:
        frame_id: The frame ID or index.
        class_id: The class label.
        bbox: The bounding box.
        confidence: The confidence score.
        polygon: The polygon resulted from instance segmentation models.
        feature: The feature used in deep tracking methods.
        timestamp: The timestamp when the detection is created.
    """
    
    def __init__(
        self,
        frame_id  : int,
        class_id  : int,
        bbox      : np.ndarray,
        confidence: float,
        polygon   : np.ndarray  = None,
        feature   : np.ndarray  = None,
        timestamp : int | float = timer(),
    ):
        self.frame_id   = frame_id
        self.class_id   = class_id
        self.bbox       = bbox
        self.confidence = confidence
        self.polygon    = np.array(polygon) if polygon else None
        self.feature    = np.array(feature) if feature else None
        self.timestamp  = timestamp
    
    @classmethod
    def from_value(cls, value: Detection | dict) -> Detection:
        """Create a `BBoxAnnotation` object from an arbitrary `value`.
        """
        if isinstance(value, dict):
            return Detection(**value)
        elif isinstance(value, Detection):
            return value
        else:
            raise ValueError(f"`value` must be a `Detection` class or a `dict`, got {type(value)}.")
    
    @property
    def bbox(self) -> np.ndarray:
        """Return the bounding box of shape [4]."""
        return self._bbox
    
    @bbox.setter
    def bbox(self, bbox: np.ndarray | list | tuple):
        bbox = np.ndarray(bbox) if not isinstance(bbox, np.ndarray) else bbox
        if bbox.ndim == 1 and bbox.size == 4:
            self._bbox = bbox
        else:
            raise ValueError(f"`bbox` must be a 1D array of size ``4``, got {bbox.ndim} and {bbox.size}.")
    
    @property
    def bbox_center(self) -> np.ndarray:
        return types.bbox_center(bbox=self.bbox)[0]
    
    @property
    def bbox_tl(self) -> np.ndarray:
        """The bbox's top left corner."""
        return self.bbox[0:2]
    
    @property
    def bbox_corners_points(self) -> np.ndarray:
        return types.bbox_corners_pts(bbox=self.bbox)[0]
    
    @property
    def confidence(self) -> float:
        """The confidence of the bounding box."""
        return self._confidence
    
    @confidence.setter
    def confidence(self, confidence: float):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"`confidence` must be between ``0.0`` and ``1.0``, got {confidence}.")
        self._confidence = confidence
    

class Track(ABC):
    """The base class for all tracks.
    
    Definition: A track represents the trajectory or path that an object takes
    as it moves through a sequence of frames in a video or across multiple
    sensor readings. It consists of a series of positional data points
    corresponding to the object's location at different times.
    
    Attributes:
        count: The total number of tracking objects.
    
    Args:
        id_: The unique ID of the track. Default: ``None``.
        state: The state of the track. Default: `TrackState.NEW`.
        detections: The list of detections associated with the track. Default: [].
    """
    
    count: int = 0
    
    def __init__(
        self,
        id_       : int        = None,
        state     : TrackState = TrackState.NEW,
        detections: Detection | list[Detection] = [],
    ):
        self.id_      = id_ or Track.count
        self.state    = state
        self.history  = detections
        Track.count  += 1
    
    @property
    def history(self) -> list[Detection]:
        """The history of the track."""
        return self._history
    
    @history.setter
    def history(self, detections: Detection | list[Detection]):
        detections = [detections] if not isinstance(detections, list) else detections
        if not all(isinstance(d, Detection) for d in detections):
            raise ValueError(f"`detections` must be a `list` of `Detection`, got {type(detections)}.")
        self._history = detections
    
    @staticmethod
    def next_id() -> int:
        """This function keeps track of the total number of tracking objects,
        which is also the track ID of the new tracking object.
        """
        return Track.count + 1
    
    @abstractmethod
    def update(self, *args, **kwargs):
        """Updates the state vector of the tracking object."""
        pass
    
    @abstractmethod
    def predict(self):
        """Predict the next state of the tracking object."""
        pass
        



# region Tracker

class Tracker(ABC):
    """The base class for all trackers."""
    
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements datasets and datamodules for NTIRE 2025 LLIE Challenge.

References:
	- https://codalab.lisn.upsaclay.fr/competitions/21636
"""

__all__ = [
	"NTIRE2025LLIE",
	"NTIRE2025LLIEDataModule",
]

from typing import Literal

from mon import core, vision
from mon.constants import DATA_DIR, DATAMODULES, DATASETS, Split, Task

# ----- Alias -----
ClassLabels                    = core.ClassLabels
DatapointAttributes            = core.DatapointAttributes
DepthMapAnnotation             = vision.DepthMapAnnotation
ImageAnnotation                = vision.ImageAnnotation
SemanticSegmentationAnnotation = vision.SemanticSegmentationAnnotation
VisionDataset                  = vision.VisionDataset


# ----- Dataset -----
@DATASETS.register(name="ntire_2025_llie")
class NTIRE2025LLIE(VisionDataset):
    """Loads NTIRE 2025 LLIE dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path = DATA_DIR / "ntire", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "ntire_2025_llie" if root.name != "ntire_2025_llie" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        if self.split in [Split.TRAIN]:
            patterns = [self.root / "train" / "image"]
        elif self.split in [Split.VAL]:
            patterns = [self.root / "val" / "image"]
        elif self.split in [Split.TEST]:
            patterns = [self.root / "test" / "image"]
        else:
            raise ValueError(f"[split] invalid: [{self.split}]")

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images
		
		
# ----- DataModule -----
@DATAMODULES.register(name="ntire_2025_llie")
class NTIRE2025LLIEDataModule(core.DataModule):
    """Configures NTIRE 2025 LLIE datasets for training/testing."""
    tasks: list[Task] = [Task.LLIE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = NTIRE2025LLIE(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = NTIRE2025LLIE(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = NTIRE2025LLIE(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()

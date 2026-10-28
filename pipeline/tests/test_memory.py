"""Prevent bounding-box views from retaining a full image for every polygon."""
import numpy as np
from filseg.metrics import gt_native


def test_native_mask_owns_only_its_crop():
    polygon = [[12, 12, 22, 12, 22, 22, 12, 22]]
    boxes = gt_native([polygon])
    assert len(boxes) == 1
    bbox, mask = boxes[0]
    assert mask.base is None
    assert mask.nbytes <= 121
    assert mask.dtype == np.bool_
    assert bbox == (12, 22, 12, 22)

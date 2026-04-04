import cv2
import numpy as np
from typing import List
from aligner import align_bracket

def resize_to_match(images, scale=0.5):
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    target_h = int(max(heights) * scale)
    target_w = int(max(widths) * scale)
    return [cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            for img in images]

def merge_hdr(images):
    # Step 1: Resize to half res
    images = resize_to_match(images, scale=0.5)

    # Step 2: Align all to middle exposure
    print("Aligning images...")
    images = align_bracket(images)

    # Step 3: Use OpenCV Mertens fusion — handles halos correctly
    # contrast=1, saturation=1, exposure=1 gives natural balanced result
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0
    )
    fused = merge_mertens.process(images)

    # Mertens returns float32 in range [0,1] — convert to uint8
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)
    return fused

import cv2
import numpy as np
from typing import List
from aligner import align_bracket

def resize_to_match(images, scale=0.5):
    heights = [img.shape[0] for img in images]
    widths  = [img.shape[1] for img in images]
    target_h = int(max(heights) * scale)
    target_w = int(max(widths)  * scale)
    return [cv2.resize(img, (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4) for img in images]

def get_luminance(img):
    f = img.astype(np.float32) / 255.0
    return 0.2126*f[:,:,2] + 0.7152*f[:,:,1] + 0.0722*f[:,:,0]

def merge_hdr(images):
    # Step 1: Resize
    images = resize_to_match(images, scale=0.5)

    # Step 2: Align
    print("Aligning images...")
    images = align_bracket(images)

    # Step 3: Sort by brightness — works for any number of exposures
    lums = [get_luminance(img).mean() for img in images]
    sorted_pairs  = sorted(zip(lums, images), key=lambda x: x[0])
    sorted_images = [img for _, img in sorted_pairs]
    darkest   = sorted_images[0]
    brightest = sorted_images[-1]
    print(f"Exposure range: darkest mean={lums[0]:.2f} "
          f"brightest mean={lums[-1]:.2f}")

    # Step 4: Mertens fusion as base — handles most of the blend
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0
    )
    fused = merge_mertens.process(images)
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    # Step 5: Adaptive window pull
    # Detect highlight threshold dynamically from fused image
    fused_lum = get_luminance(fused)
    p90 = float(np.percentile(fused_lum, 90))
    p10 = float(np.percentile(fused_lum, 10))

    # Highlight threshold: above 90th percentile = overexposed region
    highlight_threshold = np.clip(p90 * 0.95, 0.70, 0.88)
    highlight_mask = np.clip(
        (fused_lum - highlight_threshold) / (1.0 - highlight_threshold),
        0, 1
    )
    highlight_mask = cv2.GaussianBlur(
        highlight_mask.astype(np.float32), (0,0), 20.0)
    highlight_mask = np.clip(highlight_mask, 0, 1)[:,:,np.newaxis]

    # Shadow threshold: below 10th percentile = underexposed region
    shadow_threshold = np.clip(p10 * 1.5, 0.08, 0.28)
    shadow_mask = np.clip(
        (shadow_threshold - fused_lum) / shadow_threshold,
        0, 1
    )
    shadow_mask = cv2.GaussianBlur(
        shadow_mask.astype(np.float32), (0,0), 20.0)
    shadow_mask = np.clip(shadow_mask, 0, 1)[:,:,np.newaxis]

    print(f"Highlight threshold: {highlight_threshold:.2f} "
          f"Shadow threshold: {shadow_threshold:.2f}")

    # Step 6: Blend region-specific exposures
    result = fused.astype(np.float32)

    # Windows/highlights: use darkest exposure where it has real detail
    dark_lum = get_luminance(darkest)
    dark_valid = np.clip((dark_lum - 0.04) / 0.25, 0, 1)[:,:,np.newaxis]
    window_blend = highlight_mask * dark_valid
    result = result * (1 - window_blend) + \
             darkest.astype(np.float32) * window_blend

    # Shadows: use brightest exposure where it isn't clipped
    bright_lum = get_luminance(brightest)
    bright_valid = np.clip((0.96 - bright_lum) / 0.25, 0, 1)[:,:,np.newaxis]
    shadow_blend = shadow_mask * bright_valid
    result = result * (1 - shadow_blend) + \
             brightest.astype(np.float32) * shadow_blend

    return np.clip(result, 0, 255).astype(np.uint8)

import cv2
import numpy as np
from typing import List
from aligner import align_bracket

def resize_to_match(images, scale=0.5):
    target_h = int(max(img.shape[0] for img in images) * scale)
    target_w = int(max(img.shape[1] for img in images) * scale)
    return [cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4) for img in images]

def get_luminance(img):
    f = img.astype(np.float32) / 255.0
    return 0.2126*f[:,:,2] + 0.7152*f[:,:,1] + 0.0722*f[:,:,0]

def merge_hdr(images):
    images = resize_to_match(images, scale=0.5)
    print("Aligning images...")
    images = align_bracket(images)

    # Sort darkest to brightest
    lums = [float(get_luminance(img).mean()) for img in images]
    sorted_pairs = sorted(zip(lums, images), key=lambda x: x[0])
    sorted_images = [img for _, img in sorted_pairs]

    darkest   = sorted_images[0]    # Exposed for highlights/windows/sky
    brightest = sorted_images[-1]   # Exposed for shadows/interior
    middle    = sorted_images[len(sorted_images)//2]  # Balanced midtones

    print(f"Exposures: dark={min(lums):.3f} mid={lums[len(lums)//2]:.3f} bright={max(lums):.3f}")

    # Step 1: Mertens fusion as base
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0)
    fused = merge_mertens.process(images)
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    # Step 2: Detect regions from MIDDLE exposure
    # Middle exposure is the reference — it shows us where windows are blown
    # and where shadows are crushed, without being extreme itself
    mid_lum = get_luminance(middle)

    # Window/highlight detection from middle exposure
    # In middle exposure, windows are already blown (lum > 0.85)
    # These are exactly the areas where we need the darkest exposure
    p_hi = float(np.percentile(mid_lum, 88))
    p_lo = float(np.percentile(mid_lum, 12))

    # Highlight mask — areas blown in middle exposure
    hi_thresh = float(np.clip(p_hi, 0.70, 0.88))
    highlight_mask = np.clip((mid_lum - hi_thresh) / 0.10, 0, 1)
    highlight_mask = cv2.GaussianBlur(highlight_mask.astype(np.float32), (0,0), 30.0)
    highlight_mask = np.clip(highlight_mask, 0, 1)

    # Shadow mask — areas crushed in middle exposure
    sh_thresh = float(np.clip(p_lo, 0.06, 0.22))
    shadow_mask = np.clip((sh_thresh - mid_lum) / 0.08, 0, 1)
    shadow_mask = cv2.GaussianBlur(shadow_mask.astype(np.float32), (0,0), 30.0)
    shadow_mask = np.clip(shadow_mask, 0, 1)

    print(f"Window threshold: {hi_thresh:.2f}  Shadow threshold: {sh_thresh:.2f}")

    # Step 3: Start with fused as base
    result_lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Step 4: WINDOW PULL
    # Take ONLY L (luminance/detail) from darkest exposure
    # Keep A+B (color) from fused — prevents blue sky / color cast
    dark_lab = cv2.cvtColor(darkest, cv2.COLOR_BGR2LAB).astype(np.float32)
    dark_lum = get_luminance(darkest)
    # Only apply where darkest has actual detail (not pitch black)
    dark_has_detail = np.clip((dark_lum - 0.02) / 0.15, 0, 1)
    window_strength = highlight_mask * dark_has_detail
    # Blend L channel only
    result_lab[:,:,0] = result_lab[:,:,0] * (1 - window_strength) +                         dark_lab[:,:,0] * window_strength
    # Color stays from fused — no blue cast

    # Step 5: SHADOW LIFT
    # Take full pixel from brightest for dark interior areas
    # Brightest exposure has well-lit interior, so color is natural
    bright_lab = cv2.cvtColor(brightest, cv2.COLOR_BGR2LAB).astype(np.float32)
    bright_lum = get_luminance(brightest)
    # Only apply where brightest is not blown
    bright_not_blown = np.clip((0.95 - bright_lum) / 0.15, 0, 1)
    shadow_strength = shadow_mask * bright_not_blown
    # Blend full LAB for shadows — color from bright exposure is fine here
    result_lab[:,:,0] = result_lab[:,:,0] * (1 - shadow_strength) +                         bright_lab[:,:,0] * shadow_strength
    result_lab[:,:,1] = result_lab[:,:,1] * (1 - shadow_strength) +                         bright_lab[:,:,1] * shadow_strength
    result_lab[:,:,2] = result_lab[:,:,2] * (1 - shadow_strength) +                         bright_lab[:,:,2] * shadow_strength

    result = cv2.cvtColor(
        np.clip(result_lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )
    return result

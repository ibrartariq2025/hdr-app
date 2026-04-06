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

    # Sort by brightness
    lums = [float(get_luminance(img).mean()) for img in images]
    sorted_pairs = sorted(zip(lums, images), key=lambda x: x[0])
    sorted_images = [img for _, img in sorted_pairs]
    darkest = sorted_images[0]
    brightest = sorted_images[-1]
    print(f"Exposure range: {min(lums):.3f} to {max(lums):.3f}")

    # Mertens fusion as base
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0)
    fused = merge_mertens.process(images)
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    # Build masks
    fused_lum = get_luminance(fused)
    p85 = float(np.percentile(fused_lum, 85))
    p95 = float(np.percentile(fused_lum, 95))
    p15 = float(np.percentile(fused_lum, 15))
    p05 = float(np.percentile(fused_lum, 5))

    hi_thresh  = float(np.clip(p85 * 1.05, 0.72, 0.90))
    hi_feather = float(np.clip(p95 - p85, 0.05, 0.20))
    highlight_mask = np.clip((fused_lum - hi_thresh) / max(hi_feather, 0.05), 0, 1)
    highlight_mask = cv2.GaussianBlur(highlight_mask.astype(np.float32), (0,0), 25.0)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (0,0), 12.0)
    highlight_mask = np.clip(highlight_mask, 0, 1)

    sh_thresh  = float(np.clip(p15 * 0.95, 0.05, 0.25))
    sh_feather = float(np.clip(p15 - p05, 0.03, 0.12))
    shadow_mask = np.clip((sh_thresh - fused_lum) / max(sh_feather, 0.03), 0, 1)
    shadow_mask = cv2.GaussianBlur(shadow_mask.astype(np.float32), (0,0), 25.0)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0,0), 12.0)
    shadow_mask = np.clip(shadow_mask, 0, 1)

    result = fused.astype(np.float32)

    # WINDOW PULL FIX:
    # Take ONLY luminance from darkest exposure for window regions.
    # Keep color (A,B channels) from fused image to avoid blue cast.
    dark_lum = get_luminance(darkest)
    dark_valid = np.clip((dark_lum - 0.03) / 0.20, 0, 1)
    window_blend = highlight_mask * dark_valid

    # Convert both to LAB
    fused_lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)
    dark_lab  = cv2.cvtColor(darkest, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Blend ONLY the L channel from darkest — keep A,B from fused
    blended_lab = fused_lab.copy()
    blended_lab[:,:,0] = fused_lab[:,:,0] * (1 - window_blend) +                          dark_lab[:,:,0]  * window_blend
    # Keep fused colors — do NOT blend A or B channels for windows
    result = cv2.cvtColor(
        np.clip(blended_lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    ).astype(np.float32)

    # SHADOW LIFT: use brightest exposure for dark areas (full blend ok here)
    bright_lum = get_luminance(brightest)
    bright_valid = np.clip((0.97 - bright_lum) / 0.20, 0, 1)
    shadow_blend = (shadow_mask * bright_valid)[:,:,np.newaxis]
    result = result * (1 - shadow_blend) +              brightest.astype(np.float32) * shadow_blend

    return np.clip(result, 0, 255).astype(np.uint8)

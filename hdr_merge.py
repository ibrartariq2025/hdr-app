import cv2
import numpy as np
from typing import List
from aligner import align_bracket

def resize_to_match(images, scale=0.5):
    target_h = int(max(img.shape[0] for img in images) * scale)
    target_w = int(max(img.shape[1] for img in images) * scale)
    return [cv2.resize(img, (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4) for img in images]

def get_luminance(img):
    f = img.astype(np.float32) / 255.0
    return 0.2126*f[:,:,2] + 0.7152*f[:,:,1] + 0.0722*f[:,:,0]

def color_match(src, ref):
    """
    Match color/white balance of src to ref using LAB mean+std matching.
    This fixes cool cast when blending dark exposure into bright regions.
    """
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    result = np.zeros_like(src_lab)
    for c in range(3):
        src_mean = np.mean(src_lab[:,:,c])
        src_std  = np.std(src_lab[:,:,c])
        ref_mean = np.mean(ref_lab[:,:,c])
        ref_std  = np.std(ref_lab[:,:,c])
        if src_std < 1e-6:
            result[:,:,c] = src_lab[:,:,c]
        else:
            result[:,:,c] = (src_lab[:,:,c] - src_mean) * (ref_std / src_std) + ref_mean
    return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def build_smooth_mask(mask_raw, blur_sigma):
    """Heavily smooth mask to eliminate fringing at blend edges"""
    mask = cv2.GaussianBlur(mask_raw.astype(np.float32), (0,0), blur_sigma)
    # Second pass for extra smoothness
    mask = cv2.GaussianBlur(mask, (0,0), blur_sigma * 0.5)
    return np.clip(mask, 0, 1)

def merge_hdr(images):
    # Step 1: Resize to half res to save memory
    images = resize_to_match(images, scale=0.5)

    # Step 2: Align all to middle exposure
    print("Aligning images...")
    images = align_bracket(images)
    ref = images[len(images) // 2]

    # Step 3: Sort by brightness
    lums = [float(get_luminance(img).mean()) for img in images]
    sorted_pairs = sorted(zip(lums, images), key=lambda x: x[0])
    sorted_images = [img for _, img in sorted_pairs]
    darkest   = sorted_images[0]
    brightest = sorted_images[-1]
    print(f"Exposure range: {lums[0]:.3f} to {lums[-1]:.3f}")

    # Step 4: Color-match all exposures to middle
    # This is critical — removes white balance differences between exposures
    matched = []
    for img in images:
        matched.append(color_match(img, ref))

    # Also color-match the darkest/brightest we'll use for region blending
    darkest_matched   = color_match(darkest, ref)
    brightest_matched = color_match(brightest, ref)

    # Step 5: Mertens fusion on color-matched images
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0
    )
    fused = merge_mertens.process(matched)
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    # Step 6: Adaptive region masks based on image statistics
    fused_lum = get_luminance(fused)

    # Dynamic thresholds — adapts to any room brightness
    p85 = float(np.percentile(fused_lum, 85))
    p15 = float(np.percentile(fused_lum, 15))
    p95 = float(np.percentile(fused_lum, 95))
    p05 = float(np.percentile(fused_lum, 5))

    # Highlight mask — only very bright pixels (windows)
    # Use p85-p95 range as transition zone to avoid hard edges
    hi_thresh  = float(np.clip(p85 * 1.05, 0.72, 0.90))
    hi_feather = float(np.clip(p95 - p85, 0.05, 0.20))
    highlight_mask = np.clip((fused_lum - hi_thresh) / max(hi_feather, 0.05), 0, 1)
    # Heavy smooth — this eliminates fringing
    highlight_mask = build_smooth_mask(highlight_mask, 25.0)

    # Shadow mask — only very dark pixels
    sh_thresh  = float(np.clip(p15 * 0.95, 0.05, 0.25))
    sh_feather = float(np.clip(p15 - p05, 0.03, 0.12))
    shadow_mask = np.clip((sh_thresh - fused_lum) / max(sh_feather, 0.03), 0, 1)
    shadow_mask = build_smooth_mask(shadow_mask, 25.0)

    print(f"Highlight threshold: {hi_thresh:.2f} Shadow threshold: {sh_thresh:.2f}")

    result = fused.astype(np.float32)

    # Step 7: Window pull — use color-matched darkest exposure
    # Only where darkest exposure has actual detail (not clipped dark)
    dark_lum = get_luminance(darkest_matched)
    dark_valid = np.clip((dark_lum - 0.03) / 0.20, 0, 1)
    window_blend = (highlight_mask * dark_valid)[:,:,np.newaxis]
    result = result * (1 - window_blend) + \
             darkest_matched.astype(np.float32) * window_blend

    # Step 8: Shadow lift — use color-matched brightest exposure
    bright_lum = get_luminance(brightest_matched)
    bright_valid = np.clip((0.97 - bright_lum) / 0.20, 0, 1)
    shadow_blend = (shadow_mask * bright_valid)[:,:,np.newaxis]
    result = result * (1 - shadow_blend) + \
             brightest_matched.astype(np.float32) * shadow_blend

    return np.clip(result, 0, 255).astype(np.uint8)

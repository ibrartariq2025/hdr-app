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
    lums = [float(get_luminance(img).mean()) for img in images]
    sorted_pairs = sorted(zip(lums, images), key=lambda x: x[0])
    sorted_images = [img for _, img in sorted_pairs]
    darkest = sorted_images[0]
    brightest = sorted_images[-1]
    print(f"Exposure range: {min(lums):.3f} to {max(lums):.3f}")
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0)
    fused = merge_mertens.process(images)
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)
    fused_lum = get_luminance(fused)
    p85 = float(np.percentile(fused_lum, 85))
    p15 = float(np.percentile(fused_lum, 15))
    p95 = float(np.percentile(fused_lum, 95))
    p05 = float(np.percentile(fused_lum, 5))
    hi_thresh = float(np.clip(p85 * 1.05, 0.72, 0.90))
    hi_feather = float(np.clip(p95 - p85, 0.05, 0.20))
    highlight_mask = np.clip((fused_lum - hi_thresh) / max(hi_feather, 0.05), 0, 1)
    highlight_mask_s = cv2.GaussianBlur(highlight_mask.astype(np.float32), (0,0), 25.0)
    highlight_mask_s = cv2.GaussianBlur(highlight_mask_s, (0,0), 12.0)
    highlight_mask_s = np.clip(highlight_mask_s, 0, 1)
    sh_thresh = float(np.clip(p15 * 0.95, 0.05, 0.25))
    sh_feather = float(np.clip(p15 - p05, 0.03, 0.12))
    shadow_mask = np.clip((sh_thresh - fused_lum) / max(sh_feather, 0.03), 0, 1)
    shadow_mask_s = cv2.GaussianBlur(shadow_mask.astype(np.float32), (0,0), 25.0)
    shadow_mask_s = cv2.GaussianBlur(shadow_mask_s, (0,0), 12.0)
    shadow_mask_s = np.clip(shadow_mask_s, 0, 1)
    print(f"Highlight threshold: {hi_thresh:.2f} Shadow threshold: {sh_thresh:.2f}")
    result = fused.astype(np.float32)
    dark_lum = get_luminance(darkest)
    dark_valid = np.clip((dark_lum - 0.03) / 0.20, 0, 1)
    window_blend = (highlight_mask_s * dark_valid)[:,:,np.newaxis]
    result = result * (1 - window_blend) + darkest.astype(np.float32) * window_blend
    bright_lum = get_luminance(brightest)
    bright_valid = np.clip((0.97 - bright_lum) / 0.20, 0, 1)
    shadow_blend = (shadow_mask_s * bright_valid)[:,:,np.newaxis]
    result = result * (1 - shadow_blend) + brightest.astype(np.float32) * shadow_blend
    result_f = result / 255.0
    result_lum = get_luminance(np.clip(result, 0, 255).astype(np.uint8))
    highlight_area = np.clip((result_lum - 0.65) / 0.20, 0, 1)
    blue_excess = np.clip(result_f[:,:,0] - result_f[:,:,2] - 0.05, 0, 1)
    blue_fix = blue_excess * highlight_area * 0.6
    result_f[:,:,0] = np.clip(result_f[:,:,0] - blue_fix, 0, 1)
    result_f[:,:,1] = np.clip(result_f[:,:,1] - blue_fix * 0.3, 0, 1)
    return np.clip(result_f * 255, 0, 255).astype(np.uint8)

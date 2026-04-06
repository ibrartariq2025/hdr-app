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

def detect_ghosts(images):
    normalized = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = float(np.mean(gray)) + 1e-6
        normalized.append(gray / mean)
    stack = np.stack(normalized, axis=0)
    stddev = np.std(stack, axis=0)
    threshold = float(np.mean(stddev) + 2.0 * np.std(stddev))
    ghost_mask = (stddev > threshold).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    ghost_mask = cv2.dilate(ghost_mask, kernel)
    ghost_mask = cv2.GaussianBlur(ghost_mask, (0,0), 3.0)
    return np.clip(ghost_mask, 0, 1)

def merge_hdr(images):
    images = resize_to_match(images, scale=0.5)
    print("Aligning images...")
    images = align_bracket(images)

    lums = [float(get_luminance(img).mean()) for img in images]
    sorted_pairs = sorted(zip(lums, images), key=lambda x: x[0])
    sorted_images = [img for _, img in sorted_pairs]
    darkest   = sorted_images[0]
    brightest = sorted_images[-1]
    middle    = sorted_images[len(sorted_images)//2]
    print(f"Exposures: dark={min(lums):.3f} mid={lums[len(lums)//2]:.3f} bright={max(lums):.3f}")

    ghost_mask = detect_ghosts(images)
    not_ghost = 1.0 - ghost_mask

    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0)
    fused = merge_mertens.process(images)
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    # KEY FIX: detect blown pixels from FUSED image not middle
    # Only pixels that are ACTUALLY blown in fused result need darkest exposure
    # Use very tight threshold — only genuinely overexposed pixels
    fused_lum = get_luminance(fused)

    # Tight highlight mask — only pixels above 0.92 (genuinely blown)
    # No broad feathering — this prevents dark halos spreading into surroundings
    highlight_hard = np.clip((fused_lum - 0.92) / 0.06, 0, 1)
    # Use erosion before dilation to avoid halos
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    highlight_hard = cv2.erode(highlight_hard.astype(np.float32), kernel_sm)
    # Only small smooth blur — keep mask tight
    highlight_mask = cv2.GaussianBlur(highlight_hard, (0,0), 8.0)
    highlight_mask = np.clip(highlight_mask, 0, 1)
    # Remove ghost areas
    highlight_mask = highlight_mask * not_ghost

    # Shadow mask — from fused dark areas
    shadow_hard = np.clip((0.12 - fused_lum) / 0.08, 0, 1)
    shadow_hard = cv2.erode(shadow_hard.astype(np.float32), kernel_sm)
    shadow_mask = cv2.GaussianBlur(shadow_hard, (0,0), 15.0)
    shadow_mask = np.clip(shadow_mask, 0, 1)
    shadow_mask = shadow_mask * not_ghost

    print(f"Blown pixels: {float(np.mean(highlight_mask)):.1%}  "
          f"Shadow pixels: {float(np.mean(shadow_mask)):.1%}")

    result_lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)

    # WINDOW PULL — L channel only from darkest, keep fused color
    # Only applied to genuinely blown pixels — no dark halos
    dark_lab = cv2.cvtColor(darkest, cv2.COLOR_BGR2LAB).astype(np.float32)
    dark_lum = get_luminance(darkest)
    dark_has_detail = np.clip((dark_lum - 0.02) / 0.15, 0, 1)
    window_strength = highlight_mask * dark_has_detail
    result_lab[:,:,0] = result_lab[:,:,0] * (1 - window_strength) +                         dark_lab[:,:,0] * window_strength
    # Keep fused color channels — no cast

    # SHADOW LIFT — from brightest where not blown
    bright_lab = cv2.cvtColor(brightest, cv2.COLOR_BGR2LAB).astype(np.float32)
    bright_lum = get_luminance(brightest)
    bright_not_blown = np.clip((0.95 - bright_lum) / 0.15, 0, 1)
    shadow_strength = shadow_mask * bright_not_blown
    result_lab[:,:,0] = result_lab[:,:,0] * (1 - shadow_strength) +                         bright_lab[:,:,0] * shadow_strength
    result_lab[:,:,1] = result_lab[:,:,1] * (1 - shadow_strength) +                         bright_lab[:,:,1] * shadow_strength
    result_lab[:,:,2] = result_lab[:,:,2] * (1 - shadow_strength) +                         bright_lab[:,:,2] * shadow_strength

    result = cv2.cvtColor(
        np.clip(result_lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR)
    return result

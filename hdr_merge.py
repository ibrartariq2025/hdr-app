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

def weight_exposure(channel, sigma=0.2):
    norm = channel.astype(np.float32) / 255.0
    # Peak at 0.5 — perfectly exposed pixels get highest weight
    w = np.exp(-((norm - 0.5) ** 2) / (2 * sigma ** 2))
    # Hard penalize clipped highlights and crushed blacks
    w[norm > 0.96] = 0.0
    w[norm < 0.04] = 0.0
    return w

def weight_contrast(gray):
    gray_f = gray.astype(np.float32) / 255.0
    # Use larger kernel — less aggressive, fewer artifacts
    laplacian = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    return np.abs(laplacian) + 1e-6

def weight_saturation(bgr):
    float_img = bgr.astype(np.float32) / 255.0
    return np.std(float_img, axis=2) + 1e-6

def build_laplacian_pyramid(img, levels=5):
    pyramid = []
    current = img.copy()
    for _ in range(levels):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        lap = cv2.subtract(current, up)
        pyramid.append(lap)
        current = down
    pyramid.append(current)
    return pyramid

def build_gaussian_pyramid(img, levels=5):
    pyramid = [img]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid

def blend_laplacian(images, weight_maps, levels=5):
    n = len(images)
    lap_pyramids = [build_laplacian_pyramid(
        img.astype(np.float32), levels) for img in images]
    gauss_pyramids = [build_gaussian_pyramid(wt, levels) for wt in weight_maps]
    blended = []
    for level in range(levels + 1):
        blended_level = np.zeros_like(lap_pyramids[0][level])
        total_weight = np.zeros(
            gauss_pyramids[0][level].shape[:2] + (1,), dtype=np.float32)
        for i in range(n):
            w_level = gauss_pyramids[i][level]
            if w_level.ndim == 2:
                w_level = w_level[:, :, np.newaxis]
            blended_level += lap_pyramids[i][level] * w_level
            total_weight += w_level
        total_weight = np.maximum(total_weight, 1e-12)
        blended.append(blended_level / total_weight)
    result = blended[-1]
    for level in range(levels - 1, -1, -1):
        result = cv2.pyrUp(
            result,
            dstsize=(blended[level].shape[1], blended[level].shape[0]))
        result = cv2.add(result, blended[level])
    return np.clip(result, 0, 255).astype(np.uint8)

def merge_hdr(images):
    # Step 1: Resize to half res
    images = resize_to_match(images, scale=0.5)

    # Step 2: Align all to middle exposure
    print("Aligning images...")
    images = align_bracket(images)

    # Step 3: Compute weight maps with heavy smoothing
    all_weights = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        w_exp = weight_exposure(gray)
        w_cont = weight_contrast(gray)
        w_cont = np.clip(w_cont / (w_cont.max() + 1e-6), 0.01, 1.0)
        w_sat = weight_saturation(img)
        w_sat = np.clip(w_sat / (w_sat.max() + 1e-6), 0.01, 1.0)

        combined = (w_exp * w_cont * w_sat).astype(np.float32)

        # Heavy smoothing — this is the key fix for blotchy walls
        # Large kernel removes patch artifacts on flat surfaces like walls
        combined = cv2.GaussianBlur(combined, (61, 61), 20.0)
        combined = np.clip(combined, 1e-6, 1.0)
        all_weights.append(combined)

    # Step 4: Normalize
    weight_sum = sum(all_weights) + 1e-12
    norm_weights = [w / weight_sum for w in all_weights]

    # Step 5: Blend
    return blend_laplacian(images, norm_weights, levels=5)

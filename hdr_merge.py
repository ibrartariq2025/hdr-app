import cv2
import numpy as np
from typing import List

def resize_to_match(images):
    """Resize all images to match the smallest one"""
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    target_h = min(heights)
    target_w = min(widths)
    return [cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4) 
            for img in images]

def weight_exposure(channel, sigma=0.2):
    norm = channel.astype(np.float32) / 255.0
    return np.exp(-((norm - 0.5) ** 2) / (2 * sigma ** 2))

def weight_contrast(gray):
    gray_f = gray.astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=5)
    return np.abs(laplacian) + 1e-6

def weight_saturation(bgr):
    float_img = bgr.astype(np.float32) / 255.0
    return np.std(float_img, axis=2) + 1e-6

def build_laplacian_pyramid(img, levels=6):
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

def build_gaussian_pyramid(img, levels=6):
    pyramid = [img]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid

def blend_laplacian(images, weight_maps, levels=6):
    n = len(images)
    lap_pyramids = [build_laplacian_pyramid(img.astype(np.float32), levels) for img in images]
    gauss_pyramids = [build_gaussian_pyramid(wt, levels) for wt in weight_maps]
    blended = []
    for level in range(levels + 1):
        blended_level = np.zeros_like(lap_pyramids[0][level])
        total_weight = np.zeros(gauss_pyramids[0][level].shape[:2] + (1,), dtype=np.float32)
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
        result = cv2.pyrUp(result, dstsize=(blended[level].shape[1], blended[level].shape[0]))
        result = cv2.add(result, blended[level])
    return np.clip(result, 0, 255).astype(np.uint8)

def merge_hdr(images):
    # Fix: resize all images to same size first
    images = resize_to_match(images)
    
    all_weights = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w_exp = np.clip(weight_exposure(gray), 0.05, 1.0)
        w_cont = weight_contrast(gray)
        w_cont = np.clip(w_cont / (w_cont.max() + 1e-6), 0.01, 1.0)
        w_sat = weight_saturation(img)
        w_sat = np.clip(w_sat / (w_sat.max() + 1e-6), 0.01, 1.0)
        all_weights.append((w_exp * w_cont * w_sat).astype(np.float32))
    weight_sum = sum(all_weights) + 1e-12
    norm_weights = [w / weight_sum for w in all_weights]
    return blend_laplacian(images, norm_weights, levels=6)

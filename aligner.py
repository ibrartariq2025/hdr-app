import cv2
import numpy as np
from typing import List

def align_bracket(images: List[np.ndarray]) -> List[np.ndarray]:
    if len(images) < 2:
        return images
    ref_idx = len(images) // 2
    ref = images[ref_idx]
    scale = 0.25
    ref_small = cv2.resize(ref, (0,0), fx=scale, fy=scale)
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_gray = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX)
    aligned = [None] * len(images)
    aligned[ref_idx] = ref
    for i, img in enumerate(images):
        if i == ref_idx:
            continue
        try:
            aligned[i] = _align_ecc(img, ref, ref_gray, scale)
        except Exception as e:
            print(f"ECC failed for image {i}: {e}, trying ORB...")
            try:
                aligned[i] = _align_orb(img, ref)
            except Exception as e2:
                print(f"ORB also failed: {e2}, using original")
                aligned[i] = img
    return aligned

def _align_ecc(src, ref_full, ref_gray_small, scale):
    h_full, w_full = ref_full.shape[:2]
    src_small = cv2.resize(src, (0,0), fx=scale, fy=scale)
    src_gray = cv2.cvtColor(src_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    src_gray = cv2.normalize(src_gray, None, 0, 255, cv2.NORM_MINMAX)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-7)
    _, warp_matrix = cv2.findTransformECC(
        ref_gray_small, src_gray, warp_matrix,
        cv2.MOTION_EUCLIDEAN, criteria, None, 5
    )
    warp_matrix[0, 2] /= scale
    warp_matrix[1, 2] /= scale
    return cv2.warpAffine(
        src, warp_matrix, (w_full, h_full),
        flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT
    )

def _align_orb(src, ref):
    h, w = ref.shape[:2]
    scale = 0.5
    ref_small = cv2.resize(ref, (0,0), fx=scale, fy=scale)
    src_small = cv2.resize(src, (0,0), fx=scale, fy=scale)
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src_small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    ref_gray = clahe.apply(ref_gray)
    src_gray = clahe.apply(src_gray)
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(src_gray, None)
    if des1 is None or des2 is None:
        raise ValueError("No features detected")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        raise ValueError(f"Not enough matches: {len(good)}")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2) / scale
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2) / scale
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
    if H is None:
        raise ValueError("Homography failed")
    return cv2.warpPerspective(src, H, (w, h),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

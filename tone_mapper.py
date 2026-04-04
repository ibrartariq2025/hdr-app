import cv2
import numpy as np

def reinhard_global(hdr, key=0.18):
    lum = 0.2126*hdr[:,:,2] + 0.7152*hdr[:,:,1] + 0.0722*hdr[:,:,0]
    lum = lum.astype(np.float32)
    lw_avg = np.exp(np.mean(np.log(lum + 1e-6)))
    lum_scaled = (key / lw_avg) * lum
    lum_mapped = lum_scaled / (1.0 + lum_scaled)
    result = hdr.copy().astype(np.float32)
    safe_lum = np.maximum(lum, 1e-6)[:, :, np.newaxis]
    result = result * (lum_mapped[:, :, np.newaxis] / safe_lum)
    return np.clip(result, 0, 1)

def lift_shadows(img, lift=0.06):
    return np.clip(img + lift * (1.0 - img), 0, 1)

def compress_highlights(img, ceiling=0.92):
    mask = img > ceiling
    hi = (img - ceiling) / (1.0 - ceiling + 1e-6)
    hi = hi * hi * (3 - 2 * hi)
    compressed = np.where(mask, ceiling + (1.0 - ceiling) * hi * 0.6, img)
    return np.clip(compressed, 0, 1)

def local_contrast_enhance(img, sigma=40.0, strength=0.30):
    img_lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    L = img_lab[:, :, 0].astype(np.float32)
    blurred = cv2.GaussianBlur(L, (0, 0), sigma)
    detail = L - blurred
    boosted = np.clip(L + strength * detail, 0, 255)
    img_lab[:, :, 0] = boosted.astype(np.uint8)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

def tone_map_real_estate(hdr_linear, brightness_key=0.22):
    img = hdr_linear.astype(np.float32) / 255.0
    img = reinhard_global(img, key=brightness_key)
    img = lift_shadows(img, lift=0.06)
    img = compress_highlights(img, ceiling=0.90)
    img = local_contrast_enhance(img, sigma=40.0, strength=0.30)
    img_hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1].astype(np.float32) * 1.12, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    return np.clip(img * 255, 0, 255).astype(np.uint8)

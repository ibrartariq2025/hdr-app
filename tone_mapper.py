import cv2
import numpy as np

def tone_map_real_estate(hdr_linear, brightness_key=0.42):
    img = hdr_linear.astype(np.float32) / 255.0

    # Step 1: Aggressive brightness boost
    lum = 0.2126*img[:,:,2] + 0.7152*img[:,:,1] + 0.0722*img[:,:,0]
    lum = lum.astype(np.float32)
    lw_avg = np.exp(np.mean(np.log(lum + 1e-6)))
    lum_scaled = (brightness_key / lw_avg) * lum

    # Step 2: Use stronger Reinhard to lift midtones
    lum_mapped = lum_scaled / (0.5 + lum_scaled)
    result = img.copy()
    safe_lum = np.maximum(lum, 1e-6)[:, :, np.newaxis]
    result = result * (lum_mapped[:, :, np.newaxis] / safe_lum)
    result = np.clip(result, 0, 1)

    # Step 3: Curves adjustment — lift shadows, protect highlights
    # S-curve: deep blacks stay black, whites stay white, midtones lift
    result = np.where(result < 0.5,
                      2 * result * result,
                      1 - 2 * (1 - result) * (1 - result))
    result = np.clip(result, 0, 1)

    # Step 4: Gamma correction — brighten overall like Lightroom exposure +0.5
    result = np.power(result, 0.75)

    # Step 5: Work in LAB for precise control
    img_lab = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    L = img_lab[:, :, 0].astype(np.float32)
    A = img_lab[:, :, 1].astype(np.float32)
    B = img_lab[:, :, 2].astype(np.float32)

    # Step 6: Local contrast on L channel (clarity boost)
    blurred = cv2.GaussianBlur(L, (0, 0), 30.0)
    detail = L - blurred
    L = np.clip(L + 0.45 * detail, 0, 255)

    # Step 7: Lift blacks — set black point to 15 (airy real estate look)
    L = np.clip(L * (240.0/255.0) + 15, 0, 255)

    # Step 8: Protect highlights — compress above 230
    highlight_mask = L > 230
    L = np.where(highlight_mask, 230 + (L - 230) * 0.3, L)
    L = np.clip(L, 0, 255)

    # Step 9: Boost saturation significantly — match the vibrant colors
    A = np.clip((A - 128) * 1.25 + 128, 0, 255)
    B = np.clip((B - 128) * 1.25 + 128, 0, 255)

    img_lab[:, :, 0] = L.astype(np.uint8)
    img_lab[:, :, 1] = A.astype(np.uint8)
    img_lab[:, :, 2] = B.astype(np.uint8)
    result = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    # Step 10: Final white balance nudge — slightly cooler/cleaner whites
    result[:, :, 2] = np.clip(result[:, :, 2] * 0.97, 0, 1)  # Reduce red slightly
    result[:, :, 0] = np.clip(result[:, :, 0] * 1.02, 0, 1)  # Boost blue slightly

    return np.clip(result * 255, 0, 255).astype(np.uint8)

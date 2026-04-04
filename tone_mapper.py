import cv2
import numpy as np

def tone_map_real_estate(hdr_linear, brightness_key=0.38):
    img = hdr_linear.astype(np.float32) / 255.0

    # Step 1: Global Reinhard — natural, not aggressive
    lum = 0.2126*img[:,:,2] + 0.7152*img[:,:,1] + 0.0722*img[:,:,0]
    lum = lum.astype(np.float32)
    lw_avg = np.exp(np.mean(np.log(lum + 1e-6)))
    lum_scaled = (brightness_key / lw_avg) * lum
    lum_mapped = lum_scaled / (1.0 + lum_scaled)
    result = img.copy()
    safe_lum = np.maximum(lum, 1e-6)[:, :, np.newaxis]
    result = result * (lum_mapped[:, :, np.newaxis] / safe_lum)
    result = np.clip(result, 0, 1)

    # Step 2: Gentle gamma — lift overall brightness
    result = np.power(np.clip(result, 1e-6, 1.0), 0.82)

    # Step 3: Work in LAB for precise shadow/highlight/midtone control
    img_lab = cv2.cvtColor(
        (result * 255).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    L = img_lab[:, :, 0]  # 0-255
    A = img_lab[:, :, 1]  # colour channel
    B = img_lab[:, :, 2]  # colour channel

    # Step 4: Targeted tone curve on L channel
    # Blacks: keep deep (don't crush but don't lift too much)
    # Midtones: brighten
    # Highlights: protect whites from blowing
    L_norm = L / 255.0
    # Smooth S-curve that lifts midtones without halos
    L_new = np.where(
        L_norm < 0.15,
        L_norm * 0.9,                          # Keep blacks dark but not crushed
        np.where(
            L_norm < 0.5,
            0.135 + (L_norm - 0.15) * 1.15,   # Lift shadows/midtones
            np.where(
                L_norm < 0.85,
                0.5375 + (L_norm - 0.5) * 0.95, # Natural midtone-highlight transition
                0.8288 + (L_norm - 0.85) * 0.5  # Compress highlights gently
            )
        )
    )
    L = np.clip(L_new * 255.0, 0, 255)

    # Step 5: Very gentle local contrast — ONLY on midtones, not highlights
    # This prevents dark halos on white walls
    blurred = cv2.GaussianBlur(L, (0, 0), 60.0)
    detail = L - blurred
    # Only apply detail boost where L is in midtone range (not on whites/blacks)
    midtone_mask = np.clip(
        1.0 - np.abs(L / 255.0 - 0.5) * 3.0, 0, 1
    )
    L = np.clip(L + 0.20 * detail * midtone_mask, 0, 255)

    # Step 6: Fix color cast — remove teal, add slight warmth
    # A channel: push slightly warm (positive = red/magenta)
    A = np.clip((A - 128) * 1.05 + 129, 0, 255)  # +1 warm shift
    # B channel: push slightly yellow/warm
    B = np.clip((B - 128) * 1.05 + 129, 0, 255)  # +1 warm shift

    # Step 7: Selective saturation
    # Boost saturation in midtones, reduce in highlights (whites stay white)
    sat_mask = np.clip(1.0 - (L / 255.0) * 0.8, 0.2, 1.0)
    A = np.clip((A - 128) * (1.0 + 0.15 * sat_mask) + 128, 0, 255)
    B = np.clip((B - 128) * (1.0 + 0.15 * sat_mask) + 128, 0, 255)

    img_lab[:, :, 0] = L
    img_lab[:, :, 1] = A
    img_lab[:, :, 2] = B

    result = cv2.cvtColor(
        img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR
    ).astype(np.float32) / 255.0

    # Step 8: Fix white walls — push near-white pixels to clean white
    # Pixels that are already bright should go to pure white, not grey
    white_mask = np.min(result, axis=2, keepdims=True)
    white_mask = np.clip((white_mask - 0.75) / 0.25, 0, 1)
    result = result * (1 - white_mask * 0.3) + white_mask * 0.3 * np.ones_like(result)

    # Step 9: Deepen blacks — rich dark areas like the sofa
    dark_mask = np.max(result, axis=2, keepdims=True)
    dark_mask = np.clip(1.0 - dark_mask / 0.3, 0, 1)
    result = result * (1 - dark_mask * 0.2)

    return np.clip(result * 255, 0, 255).astype(np.uint8)

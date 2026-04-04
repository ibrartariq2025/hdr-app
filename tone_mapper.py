import cv2
import numpy as np

def tone_map_real_estate(img_uint8, brightness_key=0.42):
    """
    Clean real estate tone mapping targeting:
    - Bright white walls and ceiling
    - Deep rich blacks in dark areas
    - Natural vibrant colors
    - No halos, no grey veil
    """
    img = img_uint8.astype(np.float32) / 255.0

    # Step 1: Strong gamma lift — this is the main brightness control
    # 0.65 = roughly +1 stop, makes everything much brighter
    img = np.power(np.clip(img, 1e-6, 1.0), 0.65)

    # Step 2: Work in LAB for clean tonal control
    img_lab = cv2.cvtColor(
        (img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    L = img_lab[:, :, 0]  # Luminance 0-255
    A = img_lab[:, :, 1]
    B = img_lab[:, :, 2]

    # Step 3: Tone curve — matches real estate Lightroom edit
    # Blacks: deep but not crushed
    # Midtones: bright and open
    # Highlights: clean white, not blown
    L_norm = L / 255.0
    L_new = np.interp(L_norm, 
        # Input points
        [0.0,  0.05, 0.20, 0.40, 0.60, 0.80, 0.92, 1.0],
        # Output points — lift midtones, keep blacks dark, protect whites
        [0.0,  0.04, 0.22, 0.46, 0.66, 0.84, 0.94, 1.0]
    )
    L = np.clip(L_new * 255.0, 0, 255)

    # Step 4: Subtle clarity — only on midtones, NOT highlights
    # This prevents halos on walls near windows
    blurred = cv2.GaussianBlur(L, (0, 0), 80.0)
    detail = L - blurred
    # Mask: only apply clarity where luminance is between 20-75%
    # Zero clarity on bright highlights (walls, ceiling, windows)
    midtone_mask = np.clip(
        np.sin(np.pi * np.clip((L / 255.0 - 0.15) / 0.65, 0, 1)), 0, 1
    ) * 0.25
    L = np.clip(L + detail * midtone_mask, 0, 255)

    # Step 5: Push near-whites to clean white
    # Any pixel above 85% luminance goes to clean white — no grey walls
    white_push = np.clip((L / 255.0 - 0.82) / 0.18, 0, 1)
    L = np.clip(L + white_push * (255 - L) * 0.7, 0, 255)

    # Step 6: Deepen blacks — rich dark sofa, no muddy grey
    black_pull = np.clip(1.0 - (L / 255.0) / 0.25, 0, 1)
    L = np.clip(L * (1.0 - black_pull * 0.35), 0, 255)

    # Step 7: Saturation — vibrant but natural
    # More saturation in midtones, less in highlights (whites stay white)
    lum_norm = L / 255.0
    sat_boost = 1.0 + 0.20 * np.sin(np.pi * np.clip(lum_norm, 0, 1))
    A = np.clip((A - 128) * sat_boost + 128, 0, 255)
    B = np.clip((B - 128) * sat_boost + 128, 0, 255)

    # Step 8: Slight warmth — remove cool/teal cast, add clean neutral white
    A = np.clip(A + 1.5, 0, 255)   # +red/magenta
    B = np.clip(B + 2.0, 0, 255)   # +yellow

    img_lab[:, :, 0] = L
    img_lab[:, :, 1] = A
    img_lab[:, :, 2] = B

    result = cv2.cvtColor(
        img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR
    ).astype(np.float32) / 255.0

    return np.clip(result * 255, 0, 255).astype(np.uint8)

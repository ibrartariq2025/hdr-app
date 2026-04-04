import cv2
import numpy as np

def analyze_image(img):
    """
    Analyze image statistics to drive adaptive tone mapping.
    Returns a dict of measured properties.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    stats = {
        'mean_lum':      float(np.mean(L)),
        'median_lum':    float(np.median(L)),
        'shadow_pct':    float(np.mean(L < 0.15)),   # % of pixels in shadow
        'highlight_pct': float(np.mean(L > 0.85)),   # % of pixels blown
        'midtone_pct':   float(np.mean((L >= 0.15) & (L <= 0.85))),
        'std_lum':       float(np.std(L)),
        'p5':            float(np.percentile(L, 5)),  # Dark point
        'p95':           float(np.percentile(L, 95)), # Bright point
        'p50':           float(np.percentile(L, 50)), # Mid point
    }
    return stats

def build_adaptive_curve(stats):
    """
    Build a tone curve based on measured image statistics.
    Automatically adjusts for bright/dark/balanced rooms.
    """
    mean = stats['mean_lum']
    p5   = stats['p5']
    p95  = stats['p95']
    shadow_pct    = stats['shadow_pct']
    highlight_pct = stats['highlight_pct']

    # How much to lift shadows based on how dark the image is
    # Dark room = stronger shadow lift, bright room = less lift
    shadow_lift = np.clip(0.04 + shadow_pct * 0.25, 0.03, 0.12)

    # How much to protect highlights based on how blown they are
    # More blown = stronger compression
    highlight_compress = np.clip(0.85 - highlight_pct * 0.30, 0.70, 0.95)

    # Mid brightness target — aim for a well-exposed midpoint
    # If image is dark, push midtones up more
    mid_target = np.clip(0.50 + (0.45 - mean) * 0.60, 0.42, 0.62)

    # Build interpolation curve dynamically
    x_points = [0.00, 0.05, 0.20, 0.40, 0.60, 0.80, highlight_compress, 1.0]
    y_points = [
        0.00,
        shadow_lift,
        shadow_lift + 0.17,
        mid_target - 0.08,
        mid_target + 0.08,
        mid_target + 0.22,
        highlight_compress,
        np.clip(highlight_compress + 0.04, 0, 1.0)
    ]

    # Ensure y is monotonically increasing
    for i in range(1, len(y_points)):
        y_points[i] = max(y_points[i], y_points[i-1] + 0.01)
    y_points = [min(v, 1.0) for v in y_points]

    return x_points, y_points

def tone_map_real_estate(img_uint8):
    """
    Fully adaptive tone mapping.
    Analyzes the merged image and calculates all parameters dynamically.
    No hardcoded values — works for any room/lighting condition.
    """
    # Analyze what we're working with
    stats = analyze_image(img_uint8)
    print(f"Image stats: mean={stats['mean_lum']:.2f} "
          f"shadows={stats['shadow_pct']:.1%} "
          f"highlights={stats['highlight_pct']:.1%}")

    img = img_uint8.astype(np.float32) / 255.0

    # Step 1: Adaptive gamma
    # Dark image (mean < 0.35) gets stronger lift
    # Bright image (mean > 0.55) gets subtle lift
    gamma = np.clip(1.0 - (stats['mean_lum'] - 0.45) * 0.8, 0.60, 0.90)
    print(f"Adaptive gamma: {gamma:.2f}")
    img = np.power(np.clip(img, 1e-6, 1.0), gamma)

    # Step 2: LAB for tonal control
    img_lab = cv2.cvtColor(
        (img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).astype(np.float32)
    L = img_lab[:, :, 0]
    A = img_lab[:, :, 1]
    B = img_lab[:, :, 2]

    # Step 3: Apply adaptive tone curve
    x_pts, y_pts = build_adaptive_curve(stats)
    L_norm = L / 255.0
    L_new = np.interp(L_norm, x_pts, y_pts)
    L = np.clip(L_new * 255.0, 0, 255)

    # Step 4: Adaptive highlight compression
    # How
cat > tone_mapper.py << 'EOF'
import cv2
import numpy as np

def analyze_image(img):
    """
    Analyze image statistics to drive adaptive tone mapping.
    Returns a dict of measured properties.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    stats = {
        'mean_lum':      float(np.mean(L)),
        'median_lum':    float(np.median(L)),
        'shadow_pct':    float(np.mean(L < 0.15)),   # % of pixels in shadow
        'highlight_pct': float(np.mean(L > 0.85)),   # % of pixels blown
        'midtone_pct':   float(np.mean((L >= 0.15) & (L <= 0.85))),
        'std_lum':       float(np.std(L)),
        'p5':            float(np.percentile(L, 5)),  # Dark point
        'p95':           float(np.percentile(L, 95)), # Bright point
        'p50':           float(np.percentile(L, 50)), # Mid point
    }
    return stats

def build_adaptive_curve(stats):
    """
    Build a tone curve based on measured image statistics.
    Automatically adjusts for bright/dark/balanced rooms.
    """
    mean = stats['mean_lum']
    p5   = stats['p5']
    p95  = stats['p95']
    shadow_pct    = stats['shadow_pct']
    highlight_pct = stats['highlight_pct']

    # How much to lift shadows based on how dark the image is
    # Dark room = stronger shadow lift, bright room = less lift
    shadow_lift = np.clip(0.04 + shadow_pct * 0.25, 0.03, 0.12)

    # How much to protect highlights based on how blown they are
    # More blown = stronger compression
    highlight_compress = np.clip(0.85 - highlight_pct * 0.30, 0.70, 0.95)

    # Mid brightness target — aim for a well-exposed midpoint
    # If image is dark, push midtones up more
    mid_target = np.clip(0.50 + (0.45 - mean) * 0.60, 0.42, 0.62)

    # Build interpolation curve dynamically
    x_points = [0.00, 0.05, 0.20, 0.40, 0.60, 0.80, highlight_compress, 1.0]
    y_points = [
        0.00,
        shadow_lift,
        shadow_lift + 0.17,
        mid_target - 0.08,
        mid_target + 0.08,
        mid_target + 0.22,
        highlight_compress,
        np.clip(highlight_compress + 0.04, 0, 1.0)
    ]

    # Ensure y is monotonically increasing
    for i in range(1, len(y_points)):
        y_points[i] = max(y_points[i], y_points[i-1] + 0.01)
    y_points = [min(v, 1.0) for v in y_points]

    return x_points, y_points

def tone_map_real_estate(img_uint8):
    """
    Fully adaptive tone mapping.
    Analyzes the merged image and calculates all parameters dynamically.
    No hardcoded values — works for any room/lighting condition.
    """
    # Analyze what we're working with
    stats = analyze_image(img_uint8)
    print(f"Image stats: mean={stats['mean_lum']:.2f} "
          f"shadows={stats['shadow_pct']:.1%} "
          f"highlights={stats['highlight_pct']:.1%}")

    img = img_uint8.astype(np.float32) / 255.0

    # Step 1: Adaptive gamma
    # Dark image (mean < 0.35) gets stronger lift
    # Bright image (mean > 0.55) gets subtle lift
    gamma = np.clip(1.0 - (stats['mean_lum'] - 0.45) * 0.8, 0.60, 0.90)
    print(f"Adaptive gamma: {gamma:.2f}")
    img = np.power(np.clip(img, 1e-6, 1.0), gamma)

    # Step 2: LAB for tonal control
    img_lab = cv2.cvtColor(
        (img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).astype(np.float32)
    L = img_lab[:, :, 0]
    A = img_lab[:, :, 1]
    B = img_lab[:, :, 2]

    # Step 3: Apply adaptive tone curve
    x_pts, y_pts = build_adaptive_curve(stats)
    L_norm = L / 255.0
    L_new = np.interp(L_norm, x_pts, y_pts)
    L = np.clip(L_new * 255.0, 0, 255)

    # Step 4: Adaptive highlight compression
    # How aggressively to pull back blown highlights
    blown_threshold = np.clip(210 + (1.0 - stats['highlight_pct'] * 5) * 20, 195, 235)
    blown_mask = np.clip((L - blown_threshold) / 40.0, 0, 1)
    L = L * (1 - blown_mask) + (blown_threshold + (L - blown_threshold) * 0.20) * blown_mask
    L = np.clip(L, 0, 255)

    # Step 5: Adaptive clarity — stronger in flat low-contrast images
    contrast_strength = np.clip(0.35 - stats['std_lum'] * 0.5, 0.10, 0.35)
    blurred = cv2.GaussianBlur(L, (0, 0), 60.0)
    detail = L - blurred
    midtone_mask = np.clip(
        np.sin(np.pi * np.clip((L / 255.0 - 0.10) / 0.75, 0, 1)), 0, 1
    ) * contrast_strength
    L = np.clip(L + detail * midtone_mask, 0, 255)

    # Step 6: Adaptive white push
    # Bright rooms need less push, dark rooms need more
    white_threshold = np.clip(0.78 + stats['mean_lum'] * 0.12, 0.78, 0.90)
    white_push = np.clip((L / 255.0 - white_threshold) / (1.0 - white_threshold), 0, 1)
    L = np.clip(L + white_push * (250 - L) * 0.60, 0, 255)

    # Step 7: Adaptive black depth
    # Only deepen blacks if there are meaningful shadow areas
    if stats['shadow_pct'] > 0.05:
        black_threshold = np.clip(0.18 + stats['p5'] * 0.5, 0.12, 0.28)
        black_pull = np.clip(1.0 - (L / 255.0) / black_threshold, 0, 1)
        black_strength = np.clip(stats['shadow_pct'] * 1.2, 0.15, 0.40)
        L = np.clip(L * (1.0 - black_pull * black_strength), 0, 255)

    # Step 8: Adaptive saturation
    # Low saturation images get bigger boost
    bgr_f = img_uint8.astype(np.float32) / 255.0
    current_sat = float(np.mean(np.std(bgr_f, axis=2)))
    sat_boost_base = np.clip(1.15 + (0.12 - current_sat) * 2.0, 1.05, 1.35)
    print(f"Saturation boost: {sat_boost_base:.2f}")

    lum_norm = L / 255.0
    sat_curve = 1.0 + (sat_boost_base - 1.0) * np.sin(
        np.pi * np.clip(lum_norm, 0, 1)
    )
    A = np.clip((A - 128) * sat_curve + 128, 0, 255)
    B = np.clip((B - 128) * sat_curve + 128, 0, 255)

    # Step 9: Adaptive white balance
    # Measure actual color cast and correct it
    mean_a = float(np.mean(A)) - 128  # Positive = too red, negative = too green
    mean_b = float(np.mean(B)) - 128  # Positive = too yellow, negative = too blue

    # Correct cast — push toward neutral (0,0) but don't overdo it
    a_correction = np.clip(-mean_a * 0.4, -8, 8)
    b_correction = np.clip(-mean_b * 0.4 + 2.0, -8, 8)  # +2 slight warmth
    print(f"WB correction: A={a_correction:.1f} B={b_correction:.1f}")

    A = np.clip(A + a_correction, 0, 255)
    B = np.clip(B + b_correction, 0, 255)

    img_lab[:, :, 0] = L
    img_lab[:, :, 1] = A
    img_lab[:, :, 2] = B

    result = cv2.cvtColor(
        img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR
    ).astype(np.float32) / 255.0

    return np.clip(result * 255, 0, 255).astype(np.uint8)

import cv2
import numpy as np

def analyze_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    bgr_f = img.astype(np.float32) / 255.0
    return {
        "mean_lum":      float(np.mean(L)),
        "shadow_pct":    float(np.mean(L < 0.15)),
        "highlight_pct": float(np.mean(L > 0.85)),
        "std_lum":       float(np.std(L)),
        "p5":            float(np.percentile(L, 5)),
        "p50":           float(np.percentile(L, 50)),
        "mean_sat":      float(np.mean(np.std(bgr_f, axis=2))),
        "mean_a":        float(np.mean(lab[:,:,1])) - 128,
        "mean_b":        float(np.mean(lab[:,:,2])) - 128,
    }

def tone_map_real_estate(img_uint8):
    stats = analyze_image(img_uint8)
    print(f"Stats: mean={stats['mean_lum']:.2f} shad={stats['shadow_pct']:.1%} hi={stats['highlight_pct']:.1%}")
    img = img_uint8.astype(np.float32) / 255.0
    gamma = float(np.clip(1.0 - (stats["mean_lum"] - 0.45) * 0.9, 0.55, 0.85))
    print(f"Gamma: {gamma:.2f}")
    img = np.power(np.clip(img, 1e-6, 1.0), gamma)
    img_lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = img_lab[:, :, 0]
    A = img_lab[:, :, 1]
    B = img_lab[:, :, 2]
    shadow_lift = float(np.clip(0.05 + stats["shadow_pct"] * 0.30, 0.04, 0.14))
    mid_target  = float(np.clip(0.52 + (0.45 - stats["p50"]) * 0.50, 0.44, 0.62))
    hi_compress = float(np.clip(0.87 - stats["highlight_pct"] * 0.35, 0.68, 0.92))
    x_pts = [0.00, 0.05, 0.15, 0.35, 0.55, 0.75, hi_compress, 1.0]
    y_raw = [0.00, shadow_lift, shadow_lift+0.12, mid_target-0.10,
             mid_target+0.08, mid_target+0.22, hi_compress, min(hi_compress+0.03,1.0)]
    y_pts = [y_raw[0]]
    for i in range(1, len(y_raw)):
        y_pts.append(max(y_raw[i], y_pts[-1] + 0.008))
    y_pts = [min(v, 1.0) for v in y_pts]
    L = np.clip(np.interp(L / 255.0, x_pts, y_pts) * 255.0, 0, 255)
    blown_thresh = float(np.clip(195 + (1.0 - stats["highlight_pct"] * 4.0) * 20, 180, 218))
    blown_mask = np.clip((L - blown_thresh) / 40.0, 0, 1)
    L = L * (1 - blown_mask) + (blown_thresh + (L - blown_thresh) * 0.10) * blown_mask
    L = np.clip(L, 0, 255)
    clarity_strength = float(np.clip(0.30 - stats["std_lum"] * 0.4, 0.08, 0.30))
    blurred = cv2.GaussianBlur(L, (0, 0), 70.0)
    detail = L - blurred
    midtone_mask = np.clip(np.sin(np.pi * np.clip((L/255.0 - 0.08)/0.78, 0, 1)), 0, 1) * clarity_strength
    L = np.clip(L + detail * midtone_mask, 0, 255)
    white_thresh = float(np.clip(0.76 + stats["mean_lum"] * 0.15, 0.76, 0.91))
    white_push = np.clip((L/255.0 - white_thresh) / (1.0 - white_thresh + 1e-6), 0, 1)
    white_push = cv2.GaussianBlur(white_push.astype(np.float32), (0,0), 3.0)
    L = np.clip(L + white_push * (252 - L) * 0.55, 0, 255)
    if stats["shadow_pct"] > 0.03:
        black_thresh = float(np.clip(0.14 + stats["p5"] * 0.5, 0.08, 0.22))
        black_pull = np.clip(1.0 - (L/255.0) / (black_thresh + 1e-6), 0, 1)
        black_pull = cv2.GaussianBlur(black_pull.astype(np.float32), (0,0), 3.0)
        black_str = float(np.clip(stats["shadow_pct"] * 0.8, 0.08, 0.28))
        L = np.clip(L * (1.0 - black_pull * black_str), 0, 255)
    sat_deficit = float(np.clip(0.14 - stats["mean_sat"], 0.0, 0.10))
    sat_base = float(np.clip(1.20 + sat_deficit * 4.0, 1.15, 1.50))
    lum_norm = L / 255.0
    sat_curve = 1.0 + (sat_base - 1.0) * np.sin(np.pi * np.clip(lum_norm / 0.85, 0, 1))
    A = np.clip((A - 128) * sat_curve + 128, 0, 255)
    B = np.clip((B - 128) * sat_curve + 128, 0, 255)
    a_correction = float(np.clip(-stats["mean_a"] * 0.45, -10, 10))
    b_correction = float(np.clip(-stats["mean_b"] * 0.45 + 2.5, -10, 10))
    print(f"WB: A={a_correction:.1f} B={b_correction:.1f}")
    A = np.clip(A + a_correction, 0, 255)
    B = np.clip(B + b_correction, 0, 255)
    img_lab[:, :, 0] = L
    img_lab[:, :, 1] = A
    img_lab[:, :, 2] = B
    result = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    return np.clip(result * 255, 0, 255).astype(np.uint8)

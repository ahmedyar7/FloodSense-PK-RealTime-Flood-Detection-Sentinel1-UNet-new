import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import torch
import numpy as np
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import rasterio


# ── Resolve weights ───────────────────────────────────────────
def resolve_weights_path(weights_path=None):
    root       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    if weights_path:
        candidates.append(weights_path)
    candidates.extend([
        os.path.join(root, "models", "best_flood_model.pth"),
        os.path.join(root, "models", "best_model.pth"),
        os.path.join(os.path.dirname(__file__), "best_flood_model.pth"),
        os.path.join(os.path.dirname(__file__), "best_model.pth"),
    ])
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Model weights not found. Place best_flood_model.pth in models/"
    )


# ── Load model ────────────────────────────────────────────────
def load_flood_model(weights_path=None):
    weights_path = resolve_weights_path(weights_path)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=False)
    )
    model.eval()
    print(f"Flood model loaded from {weights_path}")
    return model


# ── Preprocess ────────────────────────────────────────────────
def normalize_image(img):
    out    = np.zeros((3, img.shape[1], img.shape[2]), dtype=np.float32)
    out[0] = (np.clip(img[4], -40, 10)  + 40) / 50.0   # VV
    out[1] = (np.clip(img[5], -45, -5)  + 45) / 40.0   # VH
    vv     = out[0] + 1e-8
    vh     = out[1] + 1e-8
    out[2] = np.clip(vh / vv, 0, 2) / 2.0               # ratio
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

def preprocess_image_tile(vv_tile):
    """
    Accepts a 256x256 VV slice and prepares a 3-ch tensor for UNet.
    """
    # Channel 0: normalized VV [-40, 10] → [0, 1]
    ch0 = (np.clip(vv_tile, -40, 10) + 40) / 50.0

    # Channel 1: VH approximation (VH ≈ VV - 6dB empirically)
    vh_approx = vv_tile - 6.0
    ch1 = (np.clip(vh_approx, -45, -5) + 45) / 40.0

    # Channel 2: VH/VV ratio in linear scale
    vv_lin = np.power(10.0, np.clip(vv_tile, -40, 0) / 10.0)
    vh_lin = np.power(10.0, np.clip(vh_approx, -45, -5) / 10.0)
    ratio  = np.clip(vh_lin / (vv_lin + 1e-8), 0.0, 2.0) / 2.0

    img_norm = np.stack([ch0, ch1, ratio], axis=-1).astype(np.float32)  # (256,256,3)

    transform = ToTensorV2()
    tensor = transform(image=img_norm)["image"].unsqueeze(0)  # (1, 3, 256, 256)
    return tensor

def predict_flood(model, image_bytes, district_name, bbox):
    """
    Runs Tiled U-Net on a GEE SAR tile.
    Supports larger images (e.g. 512x512) by slicing into 256x256 tiles.
    """
    with rasterio.open(io.BytesIO(image_bytes)) as src:
        vv_full = src.read(1).astype(np.float32)  # (H_full, W_full)

    h_full, w_full = vv_full.shape
    TILE_SIZE = 256

    # Create empty canvases for the results
    pred_prob_full = np.zeros((h_full, w_full), dtype=np.float32)

    # Tiled Inference Loop
    for y in range(0, h_full, TILE_SIZE):
        for x in range(0, w_full, TILE_SIZE):
            # Extract tile
            vv_tile = vv_full[y:y+TILE_SIZE, x:x+TILE_SIZE]

            # Handle edge cases (if image is not divisible by 256)
            if vv_tile.shape[0] != TILE_SIZE or vv_tile.shape[1] != TILE_SIZE:
                vv_tile = cv2.copyMakeBorder(
                    vv_tile, 0, TILE_SIZE-vv_tile.shape[0], 0, TILE_SIZE-vv_tile.shape[1], 
                    cv2.BORDER_CONSTANT, value=0
                )

            # Preprocess and Predict
            tensor = preprocess_image_tile(vv_tile)
            with torch.no_grad():
                output = torch.sigmoid(model(tensor))

            tile_prob = output.cpu().numpy()[0, 0]

            # ── Border Margin Fix ──
            BORDER = 8
            tile_prob[:BORDER, :] = 0.0
            tile_prob[-BORDER:, :] = 0.0
            tile_prob[:, :BORDER] = 0.0
            tile_prob[:, -BORDER:] = 0.0

            # Stitch back
            h_rem = min(TILE_SIZE, h_full - y)
            w_rem = min(TILE_SIZE, w_full - x)
            pred_prob_full[y:y+h_rem, x:x+w_rem] = tile_prob[:h_rem, :w_rem]

    # Final outputs
    pred_mask_full = pred_prob_full > 0.5
    display_arr_full = (np.clip(vv_full, -25, 0) + 25) / 25.0

    # ── Metrics ──────────────────────────────────────────────────
    total_pixels = pred_mask_full.size
    flood_pixels = int(pred_mask_full.sum())
    water_pct    = (flood_pixels / total_pixels) * 100

    lon_diff          = bbox[2] - bbox[0]
    lat_diff          = bbox[3] - bbox[1]
    district_area_km2 = lon_diff * lat_diff * 111 * 111
    affected_km2      = district_area_km2 * (water_pct / 100)

    risk_score = min(10, max(1, round(water_pct / 10)))
    risk_label = (
        "low"    if risk_score <= 3 else
        "medium" if risk_score <= 6 else
        "high"
    )

    print(f"  [Tiled] Water coverage : {water_pct:.2f}%")
    print(f"  [Tiled] Affected area  : {affected_km2:.1f} km²")

    return {
        "risk_score":         risk_score,
        "water_coverage_pct": round(water_pct, 2),
        "affected_area_km2":  round(affected_km2, 1),
        "flood_pixels":       flood_pixels,
        "pred_mask":          pred_mask_full,
        "pred_prob":          pred_prob_full,
        "display_arr":        display_arr_full,
        "settlement_risk":    risk_label,
        "confidence":         "high",
        "model_used":         "U-Net ResNet34 (Tiled Inference Mode)"
    }
import json
import os
from datetime import datetime

DATA_DIR = "data/json"
os.makedirs(DATA_DIR, exist_ok=True)

def save_json(filename, data):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def save_district_data(district_name, flood_pct, flow_info=None):
    path = os.path.join(DATA_DIR, "flood_masks.json")
    
    current_data = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                current_data = json.load(f)
            except:
                current_data = []
                
    entry = {
        "district": district_name,
        "flood_pct": flood_pct,
        "river_status": flow_info,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    current_data.append(entry)
    save_json("flood_masks.json", current_data)

def save_unet_prediction(district, metrics):
    path = os.path.join(DATA_DIR, "unet_predictions.json")
    
    current_data = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                current_data = json.load(f)
            except:
                current_data = []
                
    metrics['district'] = district
    metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # We remove numpy arrays before saving to JSON
    json_metrics = {k: v for k, v in metrics.items() if not k.endswith('_mask') and k != 'display_arr'}
    
    current_data.append(json_metrics)
    save_json("unet_predictions.json", current_data)

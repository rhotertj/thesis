import os
import sys
sys.path.append("/nfs/home/rhotertj/Code/thesis/src")
import torch
import numpy as np
import json

from lit_models import Cache
from metrics import postprocess_predictions
print("Imports loaded")
cache = Cache()
cache.load("/nfs/home/rhotertj/Code/thesis/experiments/multimodal/train/warm-star-17/val_results.pkl")
print("Cache loaded!")

ground_truths = cache.get("ground_truths")
confidences = torch.stack(cache.get("confidences", as_numpy=False)).numpy()
frame_idx = cache.get("frame_idx")
match_numbers = cache.get("match_numbers")
action_idx = cache.get("action_idx")

# offset ground truth anchor positions to avoid collisions across matches
max_frame_magnitude = len(str(frame_idx.max()))
frame_offset = 10**(max_frame_magnitude + 1)
offset_frame_idx = frame_idx + frame_offset * match_numbers

pred_order = np.argsort(offset_frame_idx)
offset_frame_idx = offset_frame_idx[pred_order]
confidences = confidences[pred_order]

gt_anchors = []
gt_labels = []
for i, action_frame in enumerate(action_idx):
    if action_frame == -1: # background action
        continue
    offset_action_frame = frame_offset * match_numbers[i] + action_frame
    # we usually see the same actions T times
    if offset_action_frame in gt_anchors:
        continue

    gt_labels.append(ground_truths[i])
    gt_anchors.append(offset_action_frame)

gt_anchors = np.array(gt_anchors)
gt_labels = np.array(gt_labels)

gt_order = np.argsort(gt_anchors)
gt_anchors = gt_anchors[gt_order]
gt_labels = gt_labels[gt_order]

pred_anchors, pred_confidences = postprocess_predictions(confidences, offset_frame_idx)

fps = 29.97
label_name = ["Background", "Pass", "Shot"]
# TODO Sample random action
label_dict = {
    "UrlLocal" : "all_games/",
    "UrlYoutube" : "",
    "annotations" : [
        {
            "gameTime": f"1 - {int(np.floor((anchor/fps) / 3600)):2}:{int((anchor/fps) % 60):2}", # hours minutes
            "label": label_name[label],
            "position": int(anchor * fps * 1000), # milliseconds
            "team": "away",
            "visibility": "visible",
            "frame" : int(anchor)
        }
        for label, anchor in zip(gt_labels, gt_anchors)
    ]
}

label_dict = json.dumps(label_dict)
label_dict = json.loads(label_dict)


with open("/nfs/home/rhotertj/Code/SoccerNetv2-DevKit/data/labels/all_games/Labels-v2.json", "w+") as f:
    json.dump(label_dict, f)

pred_dict = {
    "UrlLocal": "all_games/",
    "predictions" : [
        {
            "gameTime": f"1 - {int(np.floor((anchor/fps) / 60)):2}:{int((anchor/fps) % 60)}", # hours minutes
            "label": label_name[confidence.argmax()],
            "position": int(anchor * fps * 1000), # milliseconds
            "team": "away",
            "confidence": float(confidence.max()),
            "half" : 1,
            "frame" : int(anchor)
        }
        for anchor, confidence in zip(pred_anchors, pred_confidences) if confidence.argmax() != 0
    ]
}

pred_dict = json.dumps(pred_dict)
pred_dict = json.loads(pred_dict)

with open("/nfs/home/rhotertj/Code/SoccerNetv2-DevKit/data/predictions/all_games/Predictions-v2.json", "w+") as f:
    json.dump(pred_dict, f)
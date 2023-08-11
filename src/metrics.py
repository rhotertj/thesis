import numpy as np
import scipy



def average_mAP(pred_confidences: np.ndarray, pred_anchors: np.ndarray, gt_labels: np.ndarray, gt_anchors: np.ndarray, tolerances: list, thresholds=np.linspace(0, 1, 200)):
    """Compute average precision for given predictions over tolerances for all classes.
    Note that ground truth anchors from different matches may have similar frame number annotations.
    This has to be taken care of before calling this function!
    
    Args:
        pred_confidences (np.ndarray): Model confidences. Shape (N,C).
        pred_anchors (np.ndarray): Frame number attributed with predictions. Shape (N,C).
        gt_labels (np.ndarray): Ground truth labels. Shape (M,).
        gt_anchors (np.ndarray): Frame number attributed with ground truth labels (M,).
        tolerances (list): Temporal tolerances (deltas).

    Returns:
        tuple: mAP per tolerance and AP per class.
    """      
    # let confidences and their times be already postprocessed
    if len(pred_anchors) == 0:
        return [0] * len(tolerances)
    
    map_per_tolerance = []
    for delta in tolerances:
        # skip background class `0`
        ap_per_class = []
        for c in range(1, pred_confidences.shape[-1]):
            # filter predictions and labels for class c
            pred_confidences_c = pred_confidences[pred_confidences.argmax(-1) == c]
            pred_anchors_c = pred_anchors[pred_confidences.argmax(-1) == c]
            gt_anchors_c = gt_anchors[gt_labels == c]
            
            rps = []
            # consider 11 point interpolation 
            # https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/#Record-every-detection-along-with-the-Confidence-score
            for threshold in thresholds:
                # only keep predictions with confidence above threshold
                pred_anchors_threshold = pred_anchors_c[pred_confidences_c[:, c] >= threshold]
                
                # create bounds for current delta
                window_lb = gt_anchors_c - int(delta / 2)
                window_ub = gt_anchors_c + int(delta / 2)
                
                # Iterate over predicted action and check whether predictions lie within a window wrt. delta and gt label.
                # If it does (windows_hit), keep index of prediction as a TP example. 
                # Note that we only check for the correct class since we filtered labels and predictions for class c.

                # If a prediction hits 2 windows, we attribute the prediction to the closest ground truth anchor.
                # If a prediction does not hit any window, we keep the index as a FP example.

                tp_idx = []
                fp_idx = []
                tp_windows_idx = set()
                for i, pred_anchor in enumerate(pred_anchors_threshold):
                    windows_hit = (pred_anchor >= window_lb) & (pred_anchor <= window_ub)
                    if windows_hit.any():
                        
                        gt_anchors_hit_idx = np.where(windows_hit)[0]
                        # multiple windows hit by one prediction
                        if windows_hit.sum() > 1:
                            # only mark closest window as hit
                            pred_gt_dist = np.absolute((gt_anchors_c[gt_anchors_hit_idx] - pred_anchor))
                            closest_idx = pred_gt_dist.argmin()
                            # only add tp prediction if window is hit for the first time
                            if not gt_anchors_hit_idx[closest_idx] in tp_windows_idx:
                                tp_idx.append(i)

                            tp_windows_idx.add(gt_anchors_hit_idx[closest_idx])
                        # only 1 prediction for window
                        else:
                            # only add tp prediction if window is hit for the first time
                            if not gt_anchors_hit_idx[0] in tp_windows_idx:
                                tp_idx.append(i)
                            tp_windows_idx.add(gt_anchors_hit_idx[0])
                    # prediction outside of window
                    else:
                        fp_idx.append(i)

                # false negative: windows without a prediction
                fn = len(gt_anchors_c) - len(tp_windows_idx)
                assert fn >= 0, f"{len(gt_anchors_c)=} {len(tp_windows_idx)=}"
                # true positive: windows with correct prediction
                tp = len(tp_idx)

                if len(pred_anchors_threshold) == 0:
                    p = 0
                else:
                    p = tp / len(pred_anchors_threshold)
                r = tp / len(gt_anchors_c)
                assert r*p <= 1, f"{r=} {p=}"

                rps.append((r,p))
            ps = np.array([p for (_, p) in rps] + [1])
            rs = np.array([r for (r, _) in rps] + [0])
            ap_c = np.sum((rs[:-1] - rs[1:]) * ps[:-1])
            ap_per_class.append(ap_c)
        map_per_tolerance.append(sum(ap_per_class) / len(ap_per_class))

    return map_per_tolerance

def reorder_predictions(cache):
    """Order predictions from cache by match and frame number.

    Args:
        data (Cache): Predictions from cache.

    Returns:
        tuple: Confidences, confidence frame numbers, ground truth anchors, ground truth labels
    """    
    ground_truths = cache.get("ground_truths")
    confidences = np.stack(cache.get("confidences", as_numpy=False))
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

    return confidences, offset_frame_idx, gt_anchors, gt_labels

    

def nms_peaks(confidences, frame_numbers, height=0.5, distance=8, width=12):
    """Non-maximum suppression. Attributes action to frame numbers given model confidences per frame. 
    This done by treating the confidences for each class separately as a 1D signal. We find peaks in this signal according to 
    the `height, `distance`, and `width` arguments.

    Args:
        confidences (np.ndarray): Model predictions of shape (N, C).
        frame_numbers (np.ndarray)): The accompanying frame numbers for the confidences.
        height (float): Minimum confidence for peaks.
        distance (int): Minimum distance between peaks.
        width (int): The minimum width of peaks.

    """    
    maxima_idx = []
    for c in range(1, confidences.shape[-1]):
        # 0.5 8 12
        c_idx, _ = scipy.signal.find_peaks(confidences[:, c], height=height, distance=distance, width=width)
        maxima_idx.extend(c_idx)
    anchors = frame_numbers[maxima_idx]
    anchor_confs = confidences[maxima_idx]
    correct_order = np.argsort(anchors)
    return anchors[correct_order], anchor_confs[correct_order]

if __name__ == "__main__":
    from lit_models import Cache

    val_res_name = "/nfs/home/rhotertj/Code/thesis/experiments/multimodal/mm_finetune/test_results.pkl"

    cache = Cache()
    cache.load(val_res_name)
    
    confs, confs_frames, anchors, anchor_labels = reorder_predictions(cache)

    pred_anchors, pred_confidences = nms_peaks(confs, confs_frames)

    fps = 29.97
    tolerances = [fps * i for i in range(1,6)]
    map_per_tolerance = average_mAP(pred_confidences, pred_anchors, anchor_labels, anchors, tolerances=tolerances)
    print(map_per_tolerance)
    print(sum(map_per_tolerance) / len(map_per_tolerance))
    
import numpy as np
import scipy



def recall_precision(tp: int, fp: int, fn: int):
    """Calculate Recall and Precision from number of TP, FP and FN predictions. 

    Args:
        tp (int): Number of true positive predictions.
        fp (int): Number of false positive predictions.
        fn (int): Number of false negative predictions.

    Returns:
        tuple: Recall and Precision.
    """    
    if (fp + tp == 0) or (tp + fn == 0):
        return 0, 0
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return recall, precision

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
            # consider 11 point interpolation https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/#Record-every-detection-along-with-the-Confidence-score
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
                # (NOTE) If multiple predictions hit the same window, all are kept as TP. TODO Only count one
                # Windows without prediction are FN examples.

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
                # false positive: predictions outside of any window
                fp = len(fp_idx)
                #r, p = recall_precision(tp, fp, fn)
                if len(pred_anchors_threshold) == 0:
                    p = 0
                else:
                    p = tp / len(pred_anchors_threshold)
                r = tp / len(gt_anchors_c)
                # print(f"{delta=} {threshold=} {c=}: {tp=}, {fp=}, {fn=} {r=} {p=}")
                assert r*p <= 1, f"{r=} {p=}"

                rps.append((r,p))

            # ap_c = sum([r * p for (r, p) in rps]) / len(thresholds)
            ps = np.array([p for (_, p) in rps] + [1])
            rs = np.array([r for (r, _) in rps] + [0])
            ap_c = np.sum((rs[:-1] - rs[1:]) * ps[:-1])
            ap_per_class.append(ap_c)
        map_per_tolerance.append(sum(ap_per_class) / len(ap_per_class))

    return map_per_tolerance

def postprocess_predictions(confidences :np.ndarray, frame_numbers : np.ndarray):
    """Determines predictions and their temporal anchor from sliding window predictions.

    NOTE: If multiple matches are predicted, frame numbers should be altered such that they do not fall in the same range!
    Args:
        confidences (np.ndarray): Model confidences for every sliding window position.
        frame_numbers (np.ndarray): Frame number for each prediction.

    Returns:
        tuple: Anchors and confidences.
    """    

    # get actionness over sequence
    ones = np.ones(16)
    passes = np.convolve(confidences[:, 1], ones, mode="same")
    shots = np.convolve(confidences[:, 2], ones, mode="same")
    thresh = 4

    pass4 = np.where(passes > thresh)[0]
    shot4 = np.where(shots > thresh)[0]

    pred_anchor = []
    pred_confidences = []
    current_window = []
    # aggregate subsequent action-containing frames in `current_window`
    for i in range(len(confidences) - 1):
        action_spotted = (i in pass4) or (i in shot4)
        if action_spotted:
            current_window.append(i)
        # end of action
        end_of_window = (len(current_window) > 0) and not action_spotted
        # end of frame sequence 
        end_of_sequence = (frame_numbers[i+1] - frame_numbers[i]) > 2
        if (end_of_window or end_of_sequence) and len(current_window) > 0:
            # may be more than one action in the window
            action_idx, window_confidences = predictions_from_window(current_window, confidences)
            for i, idx in enumerate(action_idx):
                pred_anchor.append(frame_numbers[idx])
                pred_confidences.append(window_confidences[i])

            current_window.clear()

        
    pred_anchor = np.array(pred_anchor)
    if len(pred_confidences) == 0 and len(pred_anchor) == 0:
        return np.array([]), np.array([])
    pred_confidences = np.stack(pred_confidences)

    return pred_anchor, pred_confidences,



def predictions_from_window(window, confidences):
    # assign class with maximum area under curve to window
    window_confidences = confidences[window]
    if len(window) < 3:
        # TODO argmax 
        # return [window[0]], [confidences[window[0]]]
        return [], []
    preds = []

    # find peaks per class
    for c in range(1, window_confidences.shape[-1]):
        c_idx, _ = scipy.signal.find_peaks(window_confidences[:, c], height=0.5, distance=12)
        if len(c_idx) > 0:
            preds.append(c_idx)
    if len(preds) > 1:
        # currently return all preds at all classes
        # TODO majority vote, highest area under curve, lenght of prediction, ...
        pass
    window_idx = []
    for pred in preds:
        window_idx.extend([window[p] for p in pred])

    return window_idx, confidences[window_idx]

def postprocess_peaks_only(confidences, frame_numbers, height, distance, width):
    maxima_idx = []
    for c in range(1, confidences.shape[-1]):
        c_idx, _ = scipy.signal.find_peaks(confidences[:, c], height=height, distance=distance, width=width)
        maxima_idx.extend(c_idx)
    anchors = frame_numbers[maxima_idx]
    anchor_confs = confidences[maxima_idx]
    correct_order = np.argsort(anchors)
    return anchors[correct_order], anchor_confs[correct_order]

if __name__ == "__main__":
    import pandas as pd
    from lit_models import Cache
    import torch

    val_res_name = "/nfs/home/rhotertj/Code/thesis/experiments/multimodal/train/warm-star-17/val_results.pkl"

    cache = Cache()
    cache.load(val_res_name)
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
    tolerances = [fps * i for i in range(1,6)]
    map_per_tolerance = average_mAP(pred_confidences, pred_anchors, gt_labels, gt_anchors, tolerances=tolerances)
    print(map_per_tolerance)
    # pred_label = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    # pred_confidences = np.array([
    #     [0, 0.3, 0],
    #     [0, 1.0, 0],
    #     [0, 0.7, 0],
    #     [0, 0.7, 0],
    #     [0, 0, 0.8],
    #     [0, 0, 1.0],
    #     [0, 0, 1.0],
    #     [0, 0, 0.8]
    # ])
    # assert (pred_confidences.argmax(-1) == pred_label).all()
    # pred_anchor = np.array([10, 20, 32, 50, 53, 70, 94, 97])
    # gt_anchor   = np.array([4, 22, 47, 72, 100])
    # gt_label = np.array([1, 1, 2, 1, 2])
    
    # deltas = 8, 16, 24
    # map_per_delta = average_mAP(pred_confidences, pred_anchor, gt_label, gt_anchor, deltas, [0.2, 0.5, 1])
    # print(map_per_delta)
    # label = [((2/24 + (1/12 + 12/27) / 3) / 2), ((7/12 + 7/27) / 2), ((7/12 + 59/144) / 2)]
    # assert np.allclose(map_per_delta, label, atol=0.001), f"{map_per_delta} {label}"
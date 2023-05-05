import numpy as np
import itertools
import pandas as pd
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
                r, p = recall_precision(tp, fp, fn)
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
    import pickle as pkl
    import pandas as pd
    val_res_name = "/nfs/home/rhotertj/Code/thesis/dataset/analysis/youthful-shadow-248/val_results.pkl"
    # val_res_name = "/nfs/home/rhotertj/Code/thesis/dataset/analysis/copper-bush-8/val_results.pkl"
    with open(val_res_name, "rb") as f:
        val_results = pkl.load(f)

    df = pd.DataFrame(val_results)
    confidences = np.concatenate(df.confidences.to_numpy())
    frame_numbers = df.frame_idx.to_numpy()
    match_numbers = df.match_number.to_numpy()
    label_offsets = df.label_offset.to_numpy()
    gt_labels = df.label.to_numpy()
    # shift frame annotations for each match into a different numerical space 
    max_frame_magnitude = len(str(frame_numbers.max()))
    print("mag", max_frame_magnitude)
    print("match numbers", np.unique(match_numbers))
    frame_offset = 10**(max_frame_magnitude + 1)
    print("off", frame_offset)
    frame_numbers = frame_numbers + (frame_offset * match_numbers)
    print(frame_numbers)
    correct_order = np.argsort(frame_numbers)
    reordered_frames = frame_numbers[correct_order]
    confidences = confidences[correct_order]
    
    

    anchors, confs = postprocess_predictions(confidences, reordered_frames)
    anchors = np.array(anchors)
    confs = np.stack(confs)

    print(anchors.shape, confs.shape)
    gt_label = gt_labels[label_offsets == 0]
    gt_anchor = frame_numbers[label_offsets == 0]

    fps = 29.97
    for var, name in zip([confs, anchors, gt_label, gt_anchor], ["metric_confs", "metric_pred_anchors", "metric_gt_label", "metric_gt_anhors"]):
        with open(name+".pkl", "wb+") as f:
            pkl.dump(var, f)
    print(average_mAP(confs, anchors, gt_label, gt_anchor, tolerances=[fps, 2*fps, 3*fps, 10*fps]))

    pred_label = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    pred_confidences = np.array([
        [0, 0.3, 0],
        [0, 1.0, 0],
        [0, 0.7, 0],
        [0, 0.7, 0],
        [0, 0, 0.8],
        [0, 0, 1.0],
        [0, 0, 1.0],
        [0, 0, 0.8]
    ])
    assert (pred_confidences.argmax(-1) == pred_label).all()
    pred_anchor = np.array([10, 20, 32, 50, 53, 70, 94, 97])
    gt_anchor   = np.array([4, 22, 47, 72, 100])
    gt_label = np.array([1, 1, 2, 1, 2])
    
    deltas = 8, 16, 24
    map_per_delta = average_mAP(pred_confidences, pred_anchor, gt_label, gt_anchor, deltas, [0.2, 0.5, 1])
    print(map_per_delta)
    label = [((2/24 + (1/12 + 12/27) / 3) / 2), ((7/12 + 7/27) / 2), ((7/12 + 59/144) / 2)]
    # assert np.allclose(map_per_delta, label, atol=0.001), f"{map_per_delta} {label}"
from sklearn import metrics  # noqa
from sklearn.metrics import f1_score  # noqa
import numpy as np  # noqa
from collections import defaultdict
from pathlib import Path


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str = str + f"| {key}: "
            for k, v in value.items():
                str = str + f" {k}={v} "
            str = str + "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key, value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


# def get_test_metrics(y_pred, y_true, img_names):
#     def get_video_metrics(image, pred, label):
#         result_dict = {}
#         new_label = []
#         new_pred = []
#         # print(image[0])
#         # print(pred.shape)
#         # print(label.shape)
#         for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
#             # 分割字符串，获取'a'和'b'的值
#             s = item[0]
#             if '\\' in s:
#                 parts = s.split('\\')
#             else:
#                 parts = s.split('/')
#             a = parts[-2]
#             b = parts[-1]

#             # 如果'a'的值还没有在字典中，添加一个新的键值对
#             if a not in result_dict:
#                 result_dict[a] = []

#             # 将'b'的值添加到'a'的列表中
#             result_dict[a].append(item)
#         image_arr = list(result_dict.values())
#         # 将字典的值转换为一个列表，得到二维数组

#         for video in image_arr:
#             pred_sum = 0
#             label_sum = 0
#             leng = 0
#             for frame in video:
#                 pred_sum += float(frame[1])
#                 label_sum += int(frame[2])
#                 leng += 1
#             new_pred.append(pred_sum / leng)
#             new_label.append(int(label_sum / leng))
#         fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
#         v_auc = metrics.auc(fpr, tpr)
#         fnr = 1 - tpr
#         v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#         return v_auc, v_eer


#     y_pred = y_pred.squeeze()
#     # auc
#     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
#     auc = metrics.auc(fpr, tpr)
#     # eer
#     fnr = 1 - tpr
#     eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#     # ap
#     ap = metrics.average_precision_score(y_true, y_pred)
#     # acc
#     prediction_class = (y_pred > 0.5).astype(int)
#     correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
#     acc = correct / len(prediction_class)
#     if type(img_names[0]) is not list:
#         # calculate video-level auc for the frame-level methods.
#         try:
#             v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
#         except Exception as e:
#             print(e)
#             v_auc=auc
#     else:
#         # video-level methods
#         v_auc=auc

#     return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}


# In metrics/utils.py


def _compute_roc_metrics(y_pred, y_true):
    """
    Core ROC computation shared by frame-level and video-level metrics.
    
    Returns:
        dict with auc, eer, eer_threshold, ap, and FPR operating point metrics,
        or None if metrics cannot be computed (single class / empty).
    """
    unique_labels = np.unique(y_true)
    if len(y_true) == 0 or len(unique_labels) < 2:
        return None

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    auc_val = metrics.auc(fpr, tpr)
    ap_val = metrics.average_precision_score(y_true, y_pred)

    # --- EER and its threshold ---
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer_val = fpr[eer_idx]
    eer_thresh = float(thresholds[eer_idx])

    result = {
        'auc': auc_val,
        'eer': eer_val,
        'eer_threshold': eer_thresh,
        'ap': ap_val,
    }

    # --- F1 scores ---
    # F1 at EER threshold (calibrated threshold)
    preds_at_eer = (y_pred >= eer_thresh).astype(int)
    result['f1_at_eer'] = float(f1_score(y_true, preds_at_eer, zero_division=0))

    # F1 at naive 0.5 threshold (production default)
    preds_at_half = (y_pred >= 0.5).astype(int)
    result['f1'] = float(f1_score(y_true, preds_at_half, zero_division=0))

    # --- FPR Operating Points ---
    # At each target FPR, find the threshold and the corresponding TPR (recall).
    # TPR = "how many fakes do we catch?" at a given false positive rate.
    # This is critical for production: you pick an acceptable FPR (e.g., 1%)
    # and read off the TPR (fake detection rate) at that operating point.
    target_fprs = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1%, 0.5%, 1%, 2%, 5%
    target_fpr_names = ['0.1pct', '0.5pct', '1pct', '2pct', '5pct']

    for target_fpr, name in zip(target_fprs, target_fpr_names):
        # Find the largest threshold where FPR <= target_fpr
        # fpr is sorted ascending, so we find the last index where fpr <= target
        valid_indices = np.where(fpr <= target_fpr)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            result[f'tpr_at_fpr{name}'] = float(tpr[idx])
            result[f'thresh_at_fpr{name}'] = float(thresholds[idx])
        else:
            # Even the loosest threshold exceeds this FPR target
            result[f'tpr_at_fpr{name}'] = 0.0
            result[f'thresh_at_fpr{name}'] = 1.0

    return result


def get_test_metrics(y_pred, y_true, img_names=None):
    """
    Calculates frame-level and, optionally, video-level metrics.
    This version is robust to single-class inputs.

    Metrics returned:
        Frame-level: acc, auc, eer, eer_threshold, ap,
                     tpr_at_fprX, thresh_at_fprX (for X in 0.1%, 0.5%, 1%, 2%, 5%)
        Video-level (if img_names provided): video_acc, video_auc, video_eer,
                     video_eer_threshold, video_ap, video_tpr_at_fprX, video_thresh_at_fprX

    Args:
        y_pred (np.ndarray): 1D array of frame-level prediction probabilities.
        y_true (np.ndarray): 1D array of frame-level ground truth labels.
        img_names (list, optional): List of frame paths. If provided, video-level
                                    metrics will be calculated by grouping frames.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred).squeeze()
    y_true = np.array(y_true).squeeze()

    metrics_dict = {}

    # --- 1. Frame-level Metrics ---
    roc_metrics = _compute_roc_metrics(y_pred, y_true)
    if roc_metrics is not None:
        metrics_dict.update(roc_metrics)
    else:
        metrics_dict.update({
            'auc': -1.0, 'eer': -1.0, 'eer_threshold': -1.0, 'ap': -1.0,
        })

    # Accuracy (always calculable, uses fixed 0.5 threshold)
    pred_class = (y_pred > 0.5).astype(int)
    correct = (pred_class == y_true).sum()
    metrics_dict['acc'] = correct / len(y_true) if len(y_true) > 0 else 0.0

    # --- 2. Video-level Metrics (if img_names provided) ---
    if img_names is not None and len(img_names) > 0:
        videos = defaultdict(lambda: {'preds': [], 'label': -1})
        for path, pred, label in zip(img_names, y_pred, y_true):
            video_id = Path(path).parent.name
            videos[video_id]['preds'].append(pred)
            if videos[video_id]['label'] == -1:
                videos[video_id]['label'] = label

        video_preds = []
        video_labels = []
        for video_id, data in videos.items():
            if not data['preds']:
                continue
            video_preds.append(np.mean(data['preds']))
            video_labels.append(data['label'])

        if len(video_labels) > 1:
            video_preds = np.array(video_preds)
            video_labels = np.array(video_labels)

            v_roc_metrics = _compute_roc_metrics(video_preds, video_labels)
            if v_roc_metrics is not None:
                # Prefix all video-level metrics with 'video_'
                for k, v in v_roc_metrics.items():
                    metrics_dict[f'video_{k}'] = v
            else:
                metrics_dict.update({
                    'video_auc': -1.0, 'video_eer': -1.0,
                    'video_eer_threshold': -1.0, 'video_ap': -1.0,
                })

            # Video accuracy
            v_pred_class = (video_preds > 0.5).astype(int)
            v_correct = (v_pred_class == video_labels).sum()
            metrics_dict['video_acc'] = v_correct / len(video_labels) if len(video_labels) > 0 else 0.0

    return metrics_dict


def metrics_at_threshold(y_pred, y_true, threshold: float) -> dict:
    """
    Compute classification metrics at a **fixed** decision threshold.

    This is the complement of :func:`get_test_metrics` which finds an
    *optimal* threshold per-split.  Here we apply a threshold chosen
    elsewhere (e.g. the in-distribution EER threshold) to evaluate how
    well the model generalises at that operating point.

    Returns:
        dict with keys: acc, f1, precision, recall, fpr, fnr, n_samples
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    if len(y_true) == 0:
        return {}

    pred_class = (y_pred >= threshold).astype(int)
    n = len(y_true)
    correct = int((pred_class == y_true).sum())

    tp = int(((pred_class == 1) & (y_true == 1)).sum())
    fp = int(((pred_class == 1) & (y_true == 0)).sum())
    fn = int(((pred_class == 0) & (y_true == 1)).sum())
    tn = int(((pred_class == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = TPR
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'acc': correct / n,
        'f1': f1,
        'precision': precision,
        'recall': recall,   # TPR
        'fpr': fpr,
        'fnr': fnr,
        'n_samples': n,
    }

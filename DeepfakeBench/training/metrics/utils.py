from sklearn import metrics  # noqa
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


def get_test_metrics(y_pred, y_true, img_names=None):
    """
    Calculates frame-level and, optionally, video-level metrics.
    This version is robust to single-class inputs.

    Args:
        y_pred (np.ndarray): 1D array of frame-level prediction probabilities.
        y_true (np.ndarray): 1D array of frame-level ground truth labels.
        img_names (list, optional): List of frame paths. If provided, video-level
                                    metrics will be calculated by grouping frames.
                                    Defaults to None.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred).squeeze()
    y_true = np.array(y_true).squeeze()

    metrics_dict = {}

    # --- 1. Frame-level Metrics (Always Calculated) ---
    # --- START OF FRAME-LEVEL FIX ---
    unique_frame_labels = np.unique(y_true)
    if len(y_true) > 0 and len(unique_frame_labels) > 1:
        # Both classes are present, calculate all metrics
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        fnr = 1 - tpr
        frame_auc = metrics.auc(fpr, tpr)
        frame_eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        frame_ap = metrics.average_precision_score(y_true, y_pred)
    else:
        # Single class or empty input, cannot calculate AUC/EER/AP
        frame_auc = -1.0
        frame_eer = -1.0
        frame_ap = -1.0

    # Accuracy can always be calculated
    pred_class = (y_pred > 0.5).astype(int)
    correct = (pred_class == y_true).sum()
    frame_acc = correct / len(y_true) if len(y_true) > 0 else 0.0

    metrics_dict.update({
        'acc': frame_acc,
        'auc': frame_auc,
        'eer': frame_eer,
        'ap': frame_ap,
    })
    # --- END OF FRAME-LEVEL FIX ---

    # --- 2. Video-level Metrics (Calculated if img_names is provided) ---
    if img_names is not None and len(img_names) > 0:
        videos = defaultdict(lambda: {'preds': [], 'label': -1})
        for path, pred, label in zip(img_names, y_pred, y_true):
            # Using Path object correctly
            video_id = Path(path).parent.name
            videos[video_id]['preds'].append(pred)
            if videos[video_id]['label'] == -1:
                videos[video_id]['label'] = label

        video_preds = []
        video_labels = []
        for video_id, data in videos.items():
            if not data['preds']: continue
            video_preds.append(np.mean(data['preds']))
            video_labels.append(data['label'])

        # --- START OF VIDEO-LEVEL FIX ---
        if len(video_labels) > 1:
            video_preds = np.array(video_preds)
            video_labels = np.array(video_labels)

            unique_video_labels = np.unique(video_labels)
            if len(unique_video_labels) > 1:
                # Both classes are present at video level
                v_fpr, v_tpr, _ = metrics.roc_curve(video_labels, video_preds, pos_label=1)
                v_fnr = 1 - v_tpr
                metrics_dict['video_auc'] = metrics.auc(v_fpr, v_tpr)
                metrics_dict['video_eer'] = v_fpr[np.nanargmin(np.absolute(v_fnr - v_fpr))]
                metrics_dict['video_ap'] = metrics.average_precision_score(video_labels, video_preds)
            else:
                # Single class at video level
                metrics_dict['video_auc'] = -1.0
                metrics_dict['video_eer'] = -1.0
                metrics_dict['video_ap'] = -1.0

            # Video accuracy can always be calculated
            v_pred_class = (video_preds > 0.5).astype(int)
            v_correct = (v_pred_class == video_labels).sum()
            metrics_dict['video_acc'] = v_correct / len(video_labels) if len(video_labels) > 0 else 0.0
        # --- END OF VIDEO-LEVEL FIX ---

    return metrics_dict

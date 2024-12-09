import argparse
import os
import sys
from typing import List, Union

import motmetrics as mm
import numpy as np
import pandas as pd
from yacs.config import CfgNode
import yaml

from config import _C as cfg_file

def detection_list_to_dict(det_list):
    keys = ["frame", "track_id", "bbox_topleft_x", "bbox_topleft_y", "bbox_width",
            "bbox_height", "conf"]
    res = {k: [] for k in keys}
    for det in det_list:
        res["frame"].append(det[0])
        res["track_id"].append(det[1])
        res["bbox_topleft_x"].append(det[2])
        res["bbox_topleft_y"].append(det[3])
        res["bbox_width"].append(det[4])
        res["bbox_height"].append(det[5])
        res["conf"].append(det[6])
    return res

def load_motchallenge_format(file_path, frame_offset=1):
    """Loads a MOTChallenge annotation txt, with frame_offset being the index of the first frame of the video"""
    res = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            line = [int(x) for x in line[:6]] + [float(x) for x in line[6:]]

            # Subtract frame offset from frame indices. If indexing starts at one, we convert
            # it to start from zero.
            line[0] -= frame_offset
            res.append(tuple(line))
    return detection_list_to_dict(res)

def to_frame_list(detections: Union[pd.DataFrame, dict], total_frames=-1):
    """Convert a dict or df describing detections to a list containing info frame-by-frame."""
    if total_frames < 0:
        total_frames = max(detections["frame"]) + 1
    frames = [[] for _ in range(total_frames)]

    for fr, tx, ty, w, h, id_ in zip(detections["frame"],
                                     detections["bbox_topleft_x"],
                                     detections["bbox_topleft_y"],
                                     detections["bbox_width"],
                                     detections["bbox_height"],
                                     detections["track_id"]):
        frames[fr].append((tx, ty, w, h, id_))
    return frames

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates of the same id in the same frame (should never happen)"""
    return df.drop_duplicates(subset=["frame", "track_id"], keep="first")

def remove_single_cam_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Remove tracks from df that only appear on one camera ('cam' column needed)"""
    subdf = df[["cam", "track_id"]].drop_duplicates()
    track_cnt = subdf[["track_id"]].groupby(["track_id"]).size()
    good_ids = track_cnt[track_cnt > 1].index
    return df[df["track_id"].isin(good_ids)]

def load_annots(paths: List[str]) -> pd.DataFrame:
    """Load one txt annot for each camera, and return them in a merged dataframe."""
    dicts = [load_motchallenge_format(path) for path in paths]
    dfs = [pd.DataFrame(d) for d in dicts]
    max_frame = 0
    for i, df in enumerate(dfs):
        df["frame"] = df["frame"].apply(lambda x: x + max_frame)
        df["cam"] = i
        max_frame = max(df["frame"])
    df = pd.concat(dfs)
    return remove_duplicates(df)

def evaluate_dfs(test_df: pd.DataFrame, pred_df: pd.DataFrame, min_iou=0.5, ignore_fp=False, name="MTMC"):
    """Evaluate MOT (or merged MTMC) predictions against the ground truth annotations."""

    acc = mm.MOTAccumulator(auto_id=True)

    total_frames = max(max(pred_df["frame"]), max(test_df["frame"])) + 1
    test_by_frame = to_frame_list(test_df, total_frames)
    pred_by_frame = to_frame_list(pred_df, total_frames)

    for gt, preds in zip(test_by_frame, pred_by_frame):
        mat_gt = np.array([x[:4] for x in gt])
        mat_pred = np.array([x[:4] for x in preds])
        iou_matrix = mm.distances.iou_matrix(mat_gt, mat_pred, 1 - min_iou)
        n, m = len(gt), len(preds)

        if ignore_fp:
            # remove preds that are unmatched (would be false positives)
            matched_gt, matched_pred = mm.lap.linear_sum_assignment(iou_matrix)
            remain_preds = set(matched_pred)
            remain_pred_idx = [-1] * m
            for i, p in enumerate(remain_preds):
                remain_pred_idx[p] = i
            m = len(remain_preds)

            # now we can create the distance matrix rigged for our matching
            iou_matrix = np.full((n, m), np.nan)
            for i_gt, i_pred in zip(matched_gt, matched_pred):
                iou_matrix[i_gt, remain_pred_idx[i_pred]] = 0.0
        else:
            remain_pred_idx = list(range(m))

        pred_ids = [x[4]
                    for i, x in enumerate(preds) if remain_pred_idx[i] >= 0]
        gt_ids = [x[4] for x in gt]
        acc.update(gt_ids, pred_ids, iou_matrix)

    metrics = mm.metrics.motchallenge_metrics
    metrics.extend(["idfp", "idfn", "idtp"])
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics, name=name)
    return summary

def formatted_summary(summary):
    mh = mm.metrics.create()
    formatters = mh.formatters
    formatters["motp"] = lambda motp: "{:.2%}".format(1 - motp)
    strsummary = mm.io.render_summary(summary, formatters=formatters,
                                      namemap=mm.io.motchallenge_metric_names)
    return strsummary

def run_evaluation(cfg: CfgNode, camera_configs):
    """Evaluate mot or mtmc results defined by a config."""

    # Setup Camera configs and Layout
    preds = cfg.METRICS.PREDICTIONS
    gt = []
        
    for _, camera_data in camera_configs.items():
        gt_path = camera_data.get('gt', None)
        gt.append(gt_path)

    if len(preds) != len(gt):
        raise Exception("Number of prediction files is different from GT files!")
    else:
        # Load preds
        pred_df = load_annots(preds)
        if len(gt) > 1 and cfg.METRICS.DROP_SINGLE_CAM_TRACKS:
            pred_df = remove_single_cam_tracks(pred_df)

        # Load gt
        test_df = load_annots(gt)

        # Calculate metrics and print them
        summary = evaluate_dfs(test_df, pred_df, min_iou=cfg.METRICS.MIN_IOU, ignore_fp=cfg.METRICS.IGNORE_FP, name="MTMC" if len(gt) > 1 else "MTSC")

        # return summary
        return formatted_summary(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MOT or MTMC results.")
    parser.add_argument("--config", help="config yaml file")
    parser.add_argument("--print_metric_info", action="store_true",
                        help="Print description of computed metrics, then exit.")
    parser.add_argument("--experimental_solver", action="store_true",
                        help="use experimental implementation instead of the motmetrics package")
    args = parser.parse_args()

    if args.print_metric_info:
        metric_info = {
            "IDF1": ("IDF1 score", "idf1"),
            "IDP": ("ID Precision", "idp"),
            "IDR": ("ID Recall", "idr"),
            "Rcll": ("Recall", "recall"),
            "Prcn": ("Precision", "precision"),
            "GT": ("num_unique (number of ground truth tracks)", "num_unique_objects"),
            "MT": ("Mostly tracked (found in >=80%)", "mostly_tracked"),
            "PT": ("Partially tracked (found in <80%, >=20%)", "partially_tracked"),
            "ML": ("Mostly lost (found in <20%)", "mostly_lost"),
            "FP": ("False positives", None),
            "FN": ("False negatives", None),
            "IDs": ("ID switches", "num_switches"),
            "FM": ("Fragmentations", "num_fragmentations"),
            "MOTA": ("Mean Object Tracking Accuracy", "mota"),
            "MOTP": ("Mean Object Tracking Precision", "motp"),
            "IDt": ("num_transfer", "num_transfer"),
            "IDa": ("num_ascend", "num_ascend"),
            "IDm": ("num_migrate", "num_migrate"),
            "idfp": ("false positives after id matching", "idfp"),
            "idfn": ("false negative after id matching", "idfn"),
            "idtp": ("true positives after id matching", "idtp"),
        }
        infos = []
        mh = mm.metrics.create()
        for k, (short, keyword) in metric_info.items():
            desc = f"\n\t{mh.metrics[keyword]['help']}" if keyword else ""
            infos.append(f"{k}: {short}{desc}")
        print("\n".join(infos))

    # # MTMC configs
    # path = 'configs\\cameras_s02_cityflow.yml'
    # mode = 'MTMC'

    # MTSC configs
    path = 'configs\\camera_s02_cityflow.yml'
    mode = 'MTSC'

    camera_configs = args.config if args.config else path

    # Load cameras data from YAML file
    with open(camera_configs, 'r') as f:
        camera_configs = yaml.safe_load(f)[mode]

    summary = run_evaluation(cfg_file, camera_configs)

    print("Evaluation results:\n" + summary + "\n")

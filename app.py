# IDOL Choreography-Part Chunking Prototype
# Colab想定:
# !pip -q install -U ultralytics opencv-python pandas tqdm librosa soundfile matplotlib gradio

import os
import cv2
import math
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO


# =========================
# Constants
# =========================

MODEL_NAME = "yolo11s-pose.pt"  # 軽量版。精度を上げたいなら yolo11s-pose.pt など
CHUNK_END_EARLY_SEC = 0.4

COCO_KP_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

SKELETON = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 6),              # shoulders
    (5, 11), (6, 12),    # torso
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# 動きの学習負荷に効きやすい末端を少し重くする
KP_WEIGHTS = {
    "left_wrist": 1.4,
    "right_wrist": 1.4,
    "left_ankle": 1.3,
    "right_ankle": 1.3,
    "left_elbow": 1.1,
    "right_elbow": 1.1,
    "left_knee": 1.1,
    "right_knee": 1.1,
    "left_shoulder": 1.0,
    "right_shoulder": 1.0,
    "left_hip": 1.0,
    "right_hip": 1.0,
    "nose": 0.5,
    "left_eye": 0.2,
    "right_eye": 0.2,
    "left_ear": 0.2,
    "right_ear": 0.2,
}


# =========================
# Basic video/audio helpers
# =========================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": total_frames / fps if fps else 0,
    }


def extract_audio_with_ffmpeg(video_path, audio_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg audio extraction failed:\n" + result.stderr[-2000:])
    return audio_path


def detect_beats_from_video(video_path, work_dir, beat_step=1):
    """
    動画音声からビート時刻を推定。
    beat_step=2にすると、細かく拾いすぎたビートを1つおきに使える。
    """
    audio_path = str(Path(work_dir) / "audio.wav")
    extract_audio_with_ffmpeg(video_path, audio_path)

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_step = max(1, int(beat_step))
    beat_times_used = np.asarray(beat_times[::beat_step], dtype=float)

    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else None
    return {
        "audio_path": audio_path,
        "tempo": tempo_value,
        "beat_times_raw": np.asarray(beat_times, dtype=float),
        "beat_times_used": beat_times_used,
    }


def find_nearest_index(values, target):
    values = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(values - float(target))))


# =========================
# Pose tracking
# =========================

def run_pose_tracking(video_path, out_csv, model_name=MODEL_NAME, frame_stride=1, conf=0.25):
    """
    YOLO pose trackingで、全フレームの人物keypointをlong形式CSVに保存。
    """
    model = YOLO(model_name)
    info = get_video_info(video_path)

    records = []
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker="botsort.yaml",
        conf=conf,
        verbose=False,
    )

    for frame_idx, r in enumerate(tqdm(results, total=info["total_frames"], desc="pose tracking")):
        if frame_idx % int(frame_stride) != 0:
            continue

        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        keypoints_xy = r.keypoints.xy.cpu().numpy()

        if r.keypoints.conf is None:
            keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=float)
        else:
            keypoints_conf = r.keypoints.conf.cpu().numpy()

        if r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(boxes_xyxy)).astype(int)

        for person_i, track_id in enumerate(track_ids):
            x1, y1, x2, y2 = boxes_xyxy[person_i]
            bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
            bbox_cx = (x1 + x2) / 2
            bbox_cy = (y1 + y2) / 2
            bbox_diag = math.sqrt(max((x2 - x1) ** 2 + (y2 - y1) ** 2, 1.0))

            for kp_i, kp_name in enumerate(COCO_KP_NAMES):
                x, y = keypoints_xy[person_i][kp_i]
                kp_conf = keypoints_conf[person_i][kp_i]
                records.append({
                    "frame_idx": int(frame_idx),
                    "time_sec": frame_idx / info["fps"] if info["fps"] else None,
                    "track_id": int(track_id),
                    "kp_index": int(kp_i),
                    "kp_name": kp_name,
                    "x": float(x),
                    "y": float(y),
                    "conf": float(kp_conf),
                    "bbox_x1": float(x1),
                    "bbox_y1": float(y1),
                    "bbox_x2": float(x2),
                    "bbox_y2": float(y2),
                    "bbox_area": float(bbox_area),
                    "bbox_cx": float(bbox_cx),
                    "bbox_cy": float(bbox_cy),
                    "bbox_diag": float(bbox_diag),
                })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return df


def rank_visible_tracks(df):
    """
    長く映っていて、keypoint信頼度が高く、bboxが大きい人物を優先。
    """
    if df.empty:
        return pd.DataFrame()

    frame_person = (
        df.groupby(["frame_idx", "track_id"])
        .agg(
            mean_kp_conf=("conf", "mean"),
            visible_kps=("conf", lambda s: int((s > 0.3).sum())),
            bbox_area=("bbox_area", "mean"),
            bbox_diag=("bbox_diag", "mean"),
        )
        .reset_index()
    )

    track_stats = (
        frame_person.groupby("track_id")
        .agg(
            frames_seen=("frame_idx", "nunique"),
            mean_kp_conf=("mean_kp_conf", "mean"),
            mean_visible_kps=("visible_kps", "mean"),
            mean_bbox_area=("bbox_area", "mean"),
            mean_bbox_diag=("bbox_diag", "mean"),
        )
        .reset_index()
    )

    track_stats["score"] = (
        track_stats["frames_seen"].fillna(0)
        * track_stats["mean_kp_conf"].fillna(0)
        * track_stats["mean_visible_kps"].fillna(0)
        * np.sqrt(track_stats["mean_bbox_area"].fillna(1).clip(lower=1))
    )
    return track_stats.sort_values("score", ascending=False)


def get_person_df_nearest(df, track_id, frame_idx, max_delta=3):
    """
    指定frameにtrack_idがいない場合、近傍フレームから拾う。
    """
    g = df[df["track_id"] == int(track_id)]
    if g.empty:
        return pd.DataFrame()
    frames = g["frame_idx"].drop_duplicates().to_numpy()
    nearest = int(frames[np.argmin(np.abs(frames - int(frame_idx)))])
    if abs(nearest - int(frame_idx)) > max_delta:
        return pd.DataFrame()
    return g[g["frame_idx"] == nearest].copy()


def draw_pose_on_frame(frame, g_person, conf_th=0.3):
    """
    frameに骨格線を描画。BGR frameを破壊的に更新して返す。
    """
    if g_person.empty:
        return frame

    g_person = g_person.sort_values("kp_index")
    pts = {}
    for _, row in g_person.iterrows():
        if float(row["conf"]) >= conf_th:
            pts[int(row["kp_index"])] = (int(row["x"]), int(row["y"]))

    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 3)

    for _, row in g_person.iterrows():
        if float(row["conf"]) >= conf_th:
            cv2.circle(frame, (int(row["x"]), int(row["y"])), 4, (0, 0, 255), -1)

    return frame


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


# =========================
# Motion intensity
# =========================

def compute_frame_motion(df, target_track_id, conf_th=0.3, smooth_window=5):
    """
    選択人物について、frameごとの運動量を計算。
    - 各keypointの前フレームからの移動距離を足す
    - bbox diagonalで正規化して、ズーム差を少し軽減
    - wrist/ankleなど末端を重くする
    """
    g = df[df["track_id"] == int(target_track_id)].copy()
    if g.empty:
        return pd.DataFrame(columns=["frame_idx", "time_sec", "motion", "motion_smooth"])

    frames = sorted(g["frame_idx"].unique())
    rows = []
    prev = None

    for f in frames:
        gf = g[g["frame_idx"] == f]
        curr = {}
        time_sec = float(gf["time_sec"].iloc[0])
        bbox_diag = float(gf["bbox_diag"].mean()) if "bbox_diag" in gf else 1.0
        bbox_diag = max(bbox_diag, 1.0)

        for _, r in gf.iterrows():
            curr[r["kp_name"]] = {
                "x": float(r["x"]),
                "y": float(r["y"]),
                "conf": float(r["conf"]),
            }

        motion_values = []
        part_motions = {}

        if prev is not None:
            for kp_name in COCO_KP_NAMES:
                if kp_name not in prev or kp_name not in curr:
                    continue
                if prev[kp_name]["conf"] < conf_th or curr[kp_name]["conf"] < conf_th:
                    continue

                dx = curr[kp_name]["x"] - prev[kp_name]["x"]
                dy = curr[kp_name]["y"] - prev[kp_name]["y"]
                dist = math.sqrt(dx * dx + dy * dy) / bbox_diag
                w = KP_WEIGHTS.get(kp_name, 1.0)
                val = dist * w
                motion_values.append(val)
                part_motions[kp_name] = val

        motion = float(np.mean(motion_values)) if motion_values else 0.0
        rows.append({
            "frame_idx": int(f),
            "time_sec": time_sec,
            "motion": motion,
            **{f"motion_{k}": v for k, v in part_motions.items()}
        })
        prev = curr

    mdf = pd.DataFrame(rows)
    if len(mdf) == 0:
        return mdf

    # スムージング
    sw = max(1, int(smooth_window))
    mdf["motion_smooth"] = mdf["motion"].rolling(sw, center=True, min_periods=1).mean()
    return mdf


def summarize_top_parts(mdf_segment, top_k=3):
    """
    セグメント内でよく動いた部位を抽出。
    """
    motion_cols = [c for c in mdf_segment.columns if c.startswith("motion_") and c != "motion_smooth"]
    if not motion_cols or len(mdf_segment) == 0:
        return ""

    sums = mdf_segment[motion_cols].sum(numeric_only=True).sort_values(ascending=False)
    name_map = {
        "left_wrist": "左手首", "right_wrist": "右手首",
        "left_elbow": "左肘", "right_elbow": "右肘",
        "left_ankle": "左足首", "right_ankle": "右足首",
        "left_knee": "左膝", "right_knee": "右膝",
        "left_shoulder": "左肩", "right_shoulder": "右肩",
        "left_hip": "左腰", "right_hip": "右腰",
        "nose": "顔",
    }

    parts = []
    for col, val in sums.head(top_k).items():
        if val <= 0:
            continue
        kp = col.replace("motion_", "")
        parts.append(name_map.get(kp, kp))
    return "・".join(parts)


# =========================
# Count table and chunking
# =========================

def build_count_table(beat_times_used, count1_sec, duration, num_counts):
    """
    Count1に近いbeatから、num_counts分のカウント列を作る。
    count_in_eightは1〜8で循環。
    """
    beat_times = np.asarray(beat_times_used, dtype=float)
    if len(beat_times) < 2:
        raise ValueError("ビート検出に失敗しました。別の動画かbeat_stepを試してください。")

    if count1_sec is None or float(count1_sec) <= 0:
        start_idx = 0
    else:
        start_idx = find_nearest_index(beat_times, float(count1_sec))

    if start_idx + int(num_counts) >= len(beat_times):
        # 足りなければ取れる分だけにする
        num_counts = max(1, len(beat_times) - start_idx - 1)

    rows = []
    for i in range(int(num_counts)):
        start_t = float(beat_times[start_idx + i])
        if start_idx + i + 1 < len(beat_times):
            end_t = float(beat_times[start_idx + i + 1])
        else:
            # 最後だけ仮に直前間隔で延長
            interval = float(np.median(np.diff(beat_times[max(0, start_idx-4): start_idx+i+1])))
            end_t = min(duration, start_t + interval)

        rows.append({
            "global_count": i + 1,
            "count_in_eight": (i % 8) + 1,
            "start_sec": start_t,
            "end_sec": end_t,
            "start_beat_idx": start_idx + i,
        })

    return pd.DataFrame(rows), start_idx, float(beat_times[start_idx])


def aggregate_motion_by_count(count_df, motion_df):
    """
    カウントごとに運動情報量を集計。
    """
    rows = []
    for _, c in count_df.iterrows():
        seg = motion_df[(motion_df["time_sec"] >= c["start_sec"]) & (motion_df["time_sec"] < c["end_sec"])]
        motion_sum = float(seg["motion_smooth"].sum()) if len(seg) else 0.0
        motion_mean = float(seg["motion_smooth"].mean()) if len(seg) else 0.0
        peak_motion = float(seg["motion_smooth"].max()) if len(seg) else 0.0
        top_parts = summarize_top_parts(seg, top_k=3)

        rows.append({
            **c.to_dict(),
            "motion_sum": motion_sum,
            "motion_mean": motion_mean,
            "peak_motion": peak_motion,
            "top_parts": top_parts,
        })

    out = pd.DataFrame(rows)
    total = out["motion_sum"].sum()
    out["motion_ratio"] = out["motion_sum"] / total if total > 0 else 0.0
    return out


CHOREO_FEATURE_KPS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


CHOREO_BONE_PAIRS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


CHOREO_ANGLE_TRIPLES = [
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_elbow", "left_shoulder", "left_hip"),
    ("right_elbow", "right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip", "left_knee"),
    ("right_shoulder", "right_hip", "right_knee"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
]


def safe_feature_name(name):
    return str(name).replace("_", "")


def choreo_feature_columns():
    cols = []
    for kp_name in CHOREO_FEATURE_KPS:
        safe_name = safe_feature_name(kp_name)
        cols.extend([
            f"pose_feat_{safe_name}_x",
            f"pose_feat_{safe_name}_y",
            f"pose_feat_{safe_name}_stdx",
            f"pose_feat_{safe_name}_stdy",
            f"pose_feat_{safe_name}_flowx",
            f"pose_feat_{safe_name}_flowy",
        ])

    for a, b in CHOREO_BONE_PAIRS:
        safe_name = f"{safe_feature_name(a)}_{safe_feature_name(b)}"
        cols.extend([
            f"pose_feat_bone_{safe_name}_dx",
            f"pose_feat_bone_{safe_name}_dy",
            f"pose_feat_bone_{safe_name}_stddx",
            f"pose_feat_bone_{safe_name}_stddy",
        ])

    for a, b, c in CHOREO_ANGLE_TRIPLES:
        safe_name = f"{safe_feature_name(a)}_{safe_feature_name(b)}_{safe_feature_name(c)}"
        cols.extend([
            f"pose_feat_angle_{safe_name}_cos",
            f"pose_feat_angle_{safe_name}_sin",
            f"pose_feat_angle_{safe_name}_stdcos",
            f"pose_feat_angle_{safe_name}_stdsin",
        ])

    return cols


def weighted_mean_std(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    weights = np.clip(weights, 0.01, None)
    weight_sum = float(weights.sum())
    mean = float((values * weights).sum() / weight_sum)
    std = float(np.sqrt(((values - mean) ** 2 * weights).sum() / weight_sum))
    return mean, std


def angle_features(p1, p2, p3):
    v1 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    v2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-9:
        return None
    cos_val = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    sin_val = float(np.clip((v1[0] * v2[1] - v1[1] * v2[0]) / denom, -1.0, 1.0))
    return cos_val, sin_val


def normalized_frame_pose(frame_df, conf_th=0.3):
    coords = {}
    for _, r in frame_df.iterrows():
        if float(r["conf"]) < conf_th:
            continue
        bbox_diag = max(float(r.get("bbox_diag", 1.0)), 1.0)
        coords[str(r["kp_name"])] = (
            (float(r["x"]) - float(r.get("bbox_cx", 0.0))) / bbox_diag,
            (float(r["y"]) - float(r.get("bbox_cy", 0.0))) / bbox_diag,
            float(r["conf"]),
        )
    return coords


def build_choreo_feature_df(count_df, pose_df, target_track_id, conf_th=0.3):
    """
    カウントごとに、ボーンの形・角度・動き方向を特徴量化する。

    座標は人物bboxの中心と対角長で正規化するため、ズームや立ち位置の差を少し抑えられる。
    """
    feature_cols = choreo_feature_columns()

    if pose_df is None or pose_df.empty:
        return pd.DataFrame([{col: 0.0 for col in feature_cols} for _ in range(len(count_df))])

    track_df = pose_df[pose_df["track_id"] == int(target_track_id)].copy()
    if track_df.empty:
        return pd.DataFrame([{col: 0.0 for col in feature_cols} for _ in range(len(count_df))])

    rows = []
    for _, c in count_df.iterrows():
        seg = track_df[
            (track_df["time_sec"] >= float(c["start_sec"])) &
            (track_df["time_sec"] < float(c["end_sec"]))
        ].copy()

        row = {col: 0.0 for col in feature_cols}
        frame_poses = []
        for _, frame_df in seg.groupby("frame_idx"):
            coords = normalized_frame_pose(frame_df, conf_th=conf_th)
            if coords:
                frame_poses.append(coords)

        if not frame_poses:
            rows.append(row)
            continue

        for kp_name in CHOREO_FEATURE_KPS:
            safe_name = safe_feature_name(kp_name)
            visible = [pose[kp_name] for pose in frame_poses if kp_name in pose]

            if not visible:
                continue

            xs = [v[0] for v in visible]
            ys = [v[1] for v in visible]
            weights = [v[2] for v in visible]
            x_mean, x_std = weighted_mean_std(xs, weights)
            y_mean, y_std = weighted_mean_std(ys, weights)
            row[f"pose_feat_{safe_name}_x"] = x_mean
            row[f"pose_feat_{safe_name}_y"] = y_mean
            row[f"pose_feat_{safe_name}_stdx"] = x_std
            row[f"pose_feat_{safe_name}_stdy"] = y_std

            first = visible[0]
            last = visible[-1]
            row[f"pose_feat_{safe_name}_flowx"] = float(last[0] - first[0])
            row[f"pose_feat_{safe_name}_flowy"] = float(last[1] - first[1])

        for a, b in CHOREO_BONE_PAIRS:
            safe_name = f"{safe_feature_name(a)}_{safe_feature_name(b)}"
            vectors = []
            weights = []
            for pose in frame_poses:
                if a not in pose or b not in pose:
                    continue
                ax, ay, aw = pose[a]
                bx, by, bw = pose[b]
                vectors.append((bx - ax, by - ay))
                weights.append(min(aw, bw))

            if not vectors:
                continue

            dx_mean, dx_std = weighted_mean_std([v[0] for v in vectors], weights)
            dy_mean, dy_std = weighted_mean_std([v[1] for v in vectors], weights)
            row[f"pose_feat_bone_{safe_name}_dx"] = dx_mean
            row[f"pose_feat_bone_{safe_name}_dy"] = dy_mean
            row[f"pose_feat_bone_{safe_name}_stddx"] = dx_std
            row[f"pose_feat_bone_{safe_name}_stddy"] = dy_std

        for a, b, c_name in CHOREO_ANGLE_TRIPLES:
            safe_name = f"{safe_feature_name(a)}_{safe_feature_name(b)}_{safe_feature_name(c_name)}"
            angles = []
            weights = []
            for pose in frame_poses:
                if a not in pose or b not in pose or c_name not in pose:
                    continue
                angle = angle_features(pose[a][:2], pose[b][:2], pose[c_name][:2])
                if angle is None:
                    continue
                angles.append(angle)
                weights.append(min(pose[a][2], pose[b][2], pose[c_name][2]))

            if not angles:
                continue

            cos_mean, cos_std = weighted_mean_std([v[0] for v in angles], weights)
            sin_mean, sin_std = weighted_mean_std([v[1] for v in angles], weights)
            row[f"pose_feat_angle_{safe_name}_cos"] = cos_mean
            row[f"pose_feat_angle_{safe_name}_sin"] = sin_mean
            row[f"pose_feat_angle_{safe_name}_stdcos"] = cos_std
            row[f"pose_feat_angle_{safe_name}_stdsin"] = sin_std

        rows.append(row)

    return pd.DataFrame(rows)


def attach_choreo_features(count_motion_df, count_df, pose_df, target_track_id):
    out = count_motion_df.reset_index(drop=True).copy()
    feature_df = build_choreo_feature_df(count_df.reset_index(drop=True), pose_df, target_track_id)
    feature_df = feature_df.astype(float).fillna(0.0).reset_index(drop=True)
    return pd.concat([out, feature_df], axis=1)


def get_choreo_feature_matrix(count_motion_df):
    feature_cols = [c for c in count_motion_df.columns if c.startswith("pose_feat_")]
    if not feature_cols:
        return None

    x = count_motion_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    if x.size == 0:
        return None

    std = x.std(axis=0)
    keep = std > 1e-6
    if not np.any(keep):
        return None

    x = x[:, keep]
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x = (x - mean) / np.clip(std, 1e-6, None)

    # 1カウント単位のノイズを少し抑える
    return pd.DataFrame(x).rolling(3, center=True, min_periods=1).mean().to_numpy()


def normalize_scores(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo = float(np.percentile(values, 10))
    hi = float(np.percentile(values, 90))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def choose_length_based_chunks(
    count_motion_df,
    target_counts_per_chunk=4,
    min_counts_per_chunk=2,
    max_counts_per_chunk=8,
    nice_end_counts=(2, 4, 6, 8),
):
    """
    ボーン特徴量が十分に取れない時のフォールバック。

    運動量は分割条件に使わず、細かさ設定に応じたカウント数と
    きりのいいカウント終端だけで分ける。
    """
    df = count_motion_df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    target_len = max(1, int(target_counts_per_chunk))
    min_len = max(1, int(min_counts_per_chunk))
    max_len = max(min_len, int(max_counts_per_chunk))

    min_k = int(math.ceil(n / max_len))
    max_k = max(1, int(math.floor(n / min_len)))
    k = max(1, int(round(n / target_len)))
    k = min(max(k, min_k), max_k)

    total_motion = float(df["motion_sum"].sum()) if "motion_sum" in df.columns else 0.0
    target_motion = total_motion / k if k > 0 else total_motion

    INF = 10**18
    dp = np.full((k + 1, n + 1), INF, dtype=float)
    back = [[None for _ in range(n + 1)] for __ in range(k + 1)]
    dp[0, 0] = 0.0

    def seg_cost(i, j):
        length = j - i
        length_cost = ((length - target_len) / max(target_len, 1)) ** 2
        end_count = int(df.iloc[j - 1]["count_in_eight"])
        nice_cost = 0.0 if (end_count in nice_end_counts or j == n) else 0.25
        return length_cost + nice_cost

    for kk in range(1, k + 1):
        for j in range(1, n + 1):
            for i in range(max(0, j - max_len), j - min_len + 1):
                if dp[kk - 1, i] >= INF:
                    continue
                cost = dp[kk - 1, i] + seg_cost(i, j)
                if cost < dp[kk, j]:
                    dp[kk, j] = cost
                    back[kk][j] = i

    if back[k][n] is None:
        segments = []
        start = 0
        while start < n:
            end = min(n, start + target_len)
            if n - end and n - end < min_len:
                end = n
            segments.append((start, end))
            start = end
        return build_chunk_df(df, segments, target_motion, method="choreo_length")

    segments = []
    kk, j = k, n
    while kk > 0:
        i = back[kk][j]
        if i is None:
            break
        segments.append((i, j))
        j = i
        kk -= 1
    segments.reverse()

    return build_chunk_df(df, segments, target_motion, method="choreo_length")


def choose_choreo_part_chunks(
    count_motion_df,
    target_counts_per_chunk=4,
    min_counts_per_chunk=2,
    max_counts_per_chunk=8,
    nice_end_counts=(2, 4, 6, 8),
):
    """
    ボーン特徴量が似ているカウントを同じ振り付けパートとしてまとめる。

    境界はカウント境界に限定し、区間内のボーン特徴のまとまりと、
    前後の特徴変化が大きい境目を優先する。
    """
    df = count_motion_df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    features = get_choreo_feature_matrix(df)
    if features is None or len(features) != n or n < 2:
        return choose_length_based_chunks(
            count_motion_df,
            target_counts_per_chunk=target_counts_per_chunk,
            min_counts_per_chunk=min_counts_per_chunk,
            max_counts_per_chunk=max_counts_per_chunk,
            nice_end_counts=nice_end_counts,
        )

    d = max(1, features.shape[1])
    raw_change = np.linalg.norm(features[1:] - features[:-1], axis=1) / math.sqrt(d)
    change_scores = normalize_scores(raw_change)
    df["choreo_change_from_prev"] = np.r_[0.0, change_scores]

    target_counts_per_chunk = max(1, int(target_counts_per_chunk))
    min_len = max(1, int(min_counts_per_chunk))
    max_len = max(min_len, int(max_counts_per_chunk))
    k = max(1, int(round(n / target_counts_per_chunk)))
    k = min(k, n)

    total_motion = float(df["motion_sum"].sum())
    target_motion = total_motion / k if k > 0 else total_motion

    sum_prefix = np.vstack([np.zeros(d), np.cumsum(features, axis=0)])
    sq_prefix = np.r_[0.0, np.cumsum(np.sum(features * features, axis=1))]

    def segment_cohesion(i, j):
        length = max(1, j - i)
        s = sum_prefix[j] - sum_prefix[i]
        sq = sq_prefix[j] - sq_prefix[i]
        sse = max(0.0, float(sq - np.dot(s, s) / length))
        return sse / (length * d)

    def boundary_reward(j):
        if j <= 0 or j >= n:
            return 0.0
        return float(change_scores[j - 1])

    INF = 10**18
    dp = np.full((k + 1, n + 1), INF, dtype=float)
    back = [[None for _ in range(n + 1)] for __ in range(k + 1)]
    dp[0, 0] = 0.0

    def seg_cost(i, j):
        length = j - i
        cohesion_cost = segment_cohesion(i, j)
        length_cost = 0.18 * ((length - target_counts_per_chunk) / max(target_counts_per_chunk, 1)) ** 2
        end_count = int(df.iloc[j - 1]["count_in_eight"])
        nice_cost = 0.0 if (end_count in nice_end_counts or j == n) else 0.2
        return cohesion_cost + length_cost + nice_cost - (1.15 * boundary_reward(j))

    for kk in range(1, k + 1):
        for j in range(1, n + 1):
            for i in range(max(0, j - max_len), j - min_len + 1):
                if dp[kk - 1, i] >= INF:
                    continue
                cost = dp[kk - 1, i] + seg_cost(i, j)
                if cost < dp[kk, j]:
                    dp[kk, j] = cost
                    back[kk][j] = i

    if back[k][n] is None:
        return choose_length_based_chunks(
            count_motion_df,
            target_counts_per_chunk=target_counts_per_chunk,
            min_counts_per_chunk=min_counts_per_chunk,
            max_counts_per_chunk=max_counts_per_chunk,
            nice_end_counts=nice_end_counts,
        )

    segments = []
    kk, j = k, n
    while kk > 0:
        i = back[kk][j]
        if i is None:
            break
        segments.append((i, j))
        j = i
        kk -= 1
    segments.reverse()

    return build_chunk_df(df, segments, target_motion, method="choreo_similarity")


def choose_motion_balanced_chunks(
    count_motion_df,
    target_counts_per_chunk=4,
    min_counts_per_chunk=2,
    max_counts_per_chunk=8,
    nice_end_counts=(2, 4, 6, 8),
):
    """
    カウント境界のみを候補にして、運動情報量がなるべく均一になるようにDPで分割。

    ポイント:
    - フレーム途中では切らない
    - Count 2/4/6/8終わりなど、きりのいい境界を優先
    - 各チャンクのmotion_sumが target_motion に近くなるようにする
    """
    df = count_motion_df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    target_counts_per_chunk = max(1, int(target_counts_per_chunk))
    min_len = max(1, int(min_counts_per_chunk))
    max_len = max(min_len, int(max_counts_per_chunk))

    # 何チャンクに分けるか。例: 16カウント / 4 = 4チャンク。
    k = max(1, int(round(n / target_counts_per_chunk)))
    k = min(k, n)

    total_motion = float(df["motion_sum"].sum())
    target_motion = total_motion / k if k > 0 else total_motion

    # prefix sum
    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(df["motion_sum"].to_numpy())

    INF = 10**18
    dp = np.full((k + 1, n + 1), INF, dtype=float)
    back = [[None for _ in range(n + 1)] for __ in range(k + 1)]
    dp[0, 0] = 0.0

    def seg_cost(i, j):
        # segment: counts i..j-1
        length = j - i
        seg_motion = prefix[j] - prefix[i]

        # 運動情報量の偏りを主目的にする
        if target_motion > 1e-9:
            motion_cost = ((seg_motion - target_motion) / target_motion) ** 2
        else:
            motion_cost = 0.0

        # 長さが target_counts_per_chunk から大きく外れすぎないように少しだけ罰則
        length_cost = 0.08 * ((length - target_counts_per_chunk) / max(target_counts_per_chunk, 1)) ** 2

        # 末尾がCount2/4/6/8以外の場合は罰則。ただし最後の終端は許す。
        end_count = int(df.iloc[j - 1]["count_in_eight"])
        nice_cost = 0.0 if (end_count in nice_end_counts or j == n) else 1.0

        return motion_cost + length_cost + nice_cost

    for kk in range(1, k + 1):
        for j in range(1, n + 1):
            for i in range(max(0, j - max_len), j - min_len + 1):
                if dp[kk - 1, i] >= INF:
                    continue
                cost = dp[kk - 1, i] + seg_cost(i, j)
                if cost < dp[kk, j]:
                    dp[kk, j] = cost
                    back[kk][j] = i

    # もし指定Kで解けなければ、Greedyにフォールバック
    if back[k][n] is None:
        return greedy_chunks(count_motion_df, target_counts_per_chunk, min_counts_per_chunk, max_counts_per_chunk, nice_end_counts)

    # 復元
    segments = []
    kk, j = k, n
    while kk > 0:
        i = back[kk][j]
        if i is None:
            break
        segments.append((i, j))
        j = i
        kk -= 1
    segments.reverse()

    return build_chunk_df(df, segments, target_motion)


def greedy_chunks(
    count_motion_df,
    target_counts_per_chunk=4,
    min_counts_per_chunk=2,
    max_counts_per_chunk=8,
    nice_end_counts=(2, 4, 6, 8),
):
    """
    DPが失敗したとき用のシンプルなGreedy。
    """
    df = count_motion_df.reset_index(drop=True).copy()
    n = len(df)
    k = max(1, int(round(n / max(1, int(target_counts_per_chunk)))))
    target_motion = float(df["motion_sum"].sum()) / k if k else float(df["motion_sum"].sum())

    segments = []
    start = 0
    acc = 0.0

    for j in range(n):
        acc += float(df.iloc[j]["motion_sum"])
        length = j - start + 1
        end_count = int(df.iloc[j]["count_in_eight"])
        can_cut = (
            length >= min_counts_per_chunk
            and (end_count in nice_end_counts or j == n - 1)
            and (acc >= target_motion or length >= max_counts_per_chunk)
        )
        if can_cut:
            segments.append((start, j + 1))
            start = j + 1
            acc = 0.0

    if start < n:
        if segments and (n - start) < min_counts_per_chunk:
            prev_start, _ = segments[-1]
            segments[-1] = (prev_start, n)
        else:
            segments.append((start, n))

    return build_chunk_df(df, segments, target_motion)


def build_chunk_df(count_df, segments, target_motion, method="motion_balanced"):
    rows = []
    for chunk_id, (i, j) in enumerate(segments, start=1):
        seg = count_df.iloc[i:j]
        motion_sum = float(seg["motion_sum"].sum())
        raw_start_sec = float(seg["start_sec"].iloc[0])
        raw_end_sec = float(seg["end_sec"].iloc[-1])
        end_sec = max(raw_start_sec + 0.05, raw_end_sec - CHUNK_END_EARLY_SEC)
        duration = float(end_sec - raw_start_sec)
        density = motion_sum / max(duration, 1e-6)

        # 主に動いた部位
        part_text = "・".join([p for p in seg["top_parts"].tolist() if isinstance(p, str) and p])
        if part_text:
            # 重複削除
            parts = []
            for p in part_text.split("・"):
                if p and p not in parts:
                    parts.append(p)
            part_text = "・".join(parts[:4])
        else:
            part_text = "不明"

        if target_motion > 1e-9:
            load_ratio = motion_sum / target_motion
        else:
            load_ratio = 0.0

        if load_ratio >= 1.25:
            load_label = "高"
        elif load_ratio <= 0.75:
            load_label = "低"
        else:
            load_label = "中"

        rows.append({
            "chunk_id": chunk_id,
            "count_range": f"{int(seg['global_count'].iloc[0])}〜{int(seg['global_count'].iloc[-1])}",
            "count_in_eight_range": f"{int(seg['count_in_eight'].iloc[0])}〜{int(seg['count_in_eight'].iloc[-1])}",
            "start_sec": round(raw_start_sec, 3),
            "end_sec": round(end_sec, 3),
            "raw_end_sec": round(raw_end_sec, 3),
            "num_counts": int(len(seg)),
            "motion_sum": round(motion_sum, 5),
            "motion_load": load_label,
            "load_ratio": round(load_ratio, 3),
            "main_parts": part_text,
            "choreo_change": round(float(seg["choreo_change_from_prev"].iloc[0]), 3) if "choreo_change_from_prev" in seg.columns else 0.0,
            "chunk_method": method,
            "reason": make_chunk_reason(seg, load_label, method=method),
        })

    return pd.DataFrame(rows)


def make_chunk_reason(seg, load_label, method="motion_balanced"):
    top_parts = [p for p in seg["top_parts"].tolist() if isinstance(p, str) and p]
    main = top_parts[0] if top_parts else "身体全体"
    n = len(seg)
    if method == "choreo_similarity":
        if "choreo_change_from_prev" in seg.columns and float(seg["choreo_change_from_prev"].iloc[0]) >= 0.65:
            return f"ボーンの動きが前のパートから大きく変わるため、{n}カウントを同じ振り付けパートとして分割。主に{main}が動く。"
        return f"ボーンの形と動きが似ているため、{n}カウントを同じ振り付けパートとしてまとめて提示。"
    if method == "choreo_length":
        return f"ボーン特徴が少ないため、細かさ設定に合わせて{n}カウントを1パートとして提示。"
    if load_label == "高":
        return f"運動量が多いため、{n}カウントで短めに分割。主に{main}が動く。"
    if load_label == "低":
        return f"運動量が少ないため、{n}カウントをまとめて提示。"
    return f"運動量が中程度のため、{n}カウントを1まとまりとして提示。"


# =========================
# Visualization
# =========================

def plot_motion_counts(count_motion_df, chunk_df, out_path):
    """
    カウントごとの運動情報量バーとチャンク境界を可視化。
    """
    df = count_motion_df.copy()
    if df.empty:
        return None

    x = np.arange(len(df))
    y = df["motion_sum"].to_numpy()

    plt.figure(figsize=(max(10, len(df) * 0.45), 4.8))
    plt.bar(x, y)
    plt.xticks(x, [str(int(c)) for c in df["count_in_eight"]])
    plt.xlabel("Count in 8")
    plt.ylabel("Motion information")
    plt.title("Count-aligned motion information and chunk boundaries")

    # chunk boundary lines
    for _, ch in chunk_df.iterrows():
        start_global = int(str(ch["count_range"]).split("〜")[0])
        end_global = int(str(ch["count_range"]).split("〜")[-1])
        start_idx = start_global - int(df["global_count"].iloc[0])
        end_idx = end_global - int(df["global_count"].iloc[0])
        plt.axvline(start_idx - 0.5, linestyle="--", linewidth=1)
        plt.axvline(end_idx + 0.5, linestyle="--", linewidth=1)
        mid = (start_idx + end_idx) / 2
        ymax = max(y) if len(y) else 1
        plt.text(mid, ymax * 1.05, f"C{int(ch['chunk_id'])}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def make_chunk_contact_sheet(video_path, pose_df, target_track_id, chunk_df, motion_df, out_path):
    """
    各チャンクの代表フレームを並べる。
    代表フレームは、そのチャンク内でmotion_smoothが最大のフレーム。
    """
    info = get_video_info(video_path)
    tiles = []

    for _, ch in chunk_df.iterrows():
        start_t = float(ch["start_sec"])
        end_t = float(ch["end_sec"])
        seg_motion = motion_df[(motion_df["time_sec"] >= start_t) & (motion_df["time_sec"] < end_t)]
        if len(seg_motion):
            frame_idx = int(seg_motion.sort_values("motion_smooth", ascending=False).iloc[0]["frame_idx"])
        else:
            frame_idx = int(((start_t + end_t) / 2) * info["fps"])

        frame = read_frame(video_path, frame_idx)
        if frame is None:
            continue

        g_person = get_person_df_nearest(pose_df, target_track_id, frame_idx, max_delta=5)
        frame = draw_pose_on_frame(frame, g_person, conf_th=0.3)

        # Resize tile
        tile_w = 320
        h, w = frame.shape[:2]
        tile_h = int(h * tile_w / max(w, 1))
        frame = cv2.resize(frame, (tile_w, tile_h))

        # label area
        label_h = 85
        canvas = np.ones((tile_h + label_h, tile_w, 3), dtype=np.uint8) * 255
        canvas[:tile_h, :, :] = frame

        label1 = f"Chunk {int(ch['chunk_id'])}: Count {ch['count_range']}"
        label2 = f"Load: {ch['motion_load']} / Parts: {ch['main_parts']}"
        label3 = f"{ch['start_sec']:.2f}s - {ch['end_sec']:.2f}s"
        cv2.putText(canvas, label1, (10, tile_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, label2[:40], (10, tile_h + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, label3, (10, tile_h + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)

    if not tiles:
        return None

    # 横に最大3枚、以降は次の行
    cols = min(3, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.uint8) * 245

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        sheet[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tile

    cv2.imwrite(out_path, sheet)
    return out_path


def make_bone_view_video(video_path, pose_df, target_track_id, out_path, conf_th=0.3):
    """
    選択されたtrack_idだけを白背景に描いた視聴用動画を作る。
    元動画と同じサイズ・fpsにして、フロント側で通常動画と切り替えやすくする。
    """
    info = get_video_info(video_path)
    width = int(info["width"])
    height = int(info["height"])
    fps = float(info["fps"] or 30)

    out_path = Path(out_path)
    silent_path = out_path.with_name(f"{out_path.stem}_silent{out_path.suffix}")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(silent_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return None

    track_df = pose_df[pose_df["track_id"] == int(target_track_id)].copy()
    grouped = {int(frame_idx): g.sort_values("kp_index") for frame_idx, g in track_df.groupby("frame_idx")}
    frame_idx = 0
    ok = True
    while ok:
        ok, _ = cap.read()
        if not ok:
            break

        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        g_person = grouped.get(frame_idx)
        if g_person is not None and not g_person.empty:
            pts = {}
            for _, row in g_person.iterrows():
                if float(row["conf"]) >= conf_th:
                    pts[int(row["kp_index"])] = (int(row["x"]), int(row["y"]))

            line_w = max(5, int(min(width, height) * 0.008))
            point_r = max(7, int(min(width, height) * 0.011))

            for a, b in SKELETON:
                if a in pts and b in pts:
                    cv2.line(canvas, pts[a], pts[b], (20, 20, 20), line_w, cv2.LINE_AA)

            for p in pts.values():
                cv2.circle(canvas, p, point_r, (255, 120, 20), -1, cv2.LINE_AA)

        writer.write(canvas)
        frame_idx += 1

    cap.release()
    writer.release()

    encoded_path = out_path.with_name(f"{out_path.stem}_encoded{out_path.suffix}")
    encode_cmd = [
        "ffmpeg", "-y",
        "-i", str(silent_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(encoded_path),
    ]
    try:
        encode_result = subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        encode_failed = encode_result.returncode != 0
    except OSError:
        encode_failed = True

    video_for_mux = silent_path if encode_failed else encoded_path

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_for_mux),
        "-i", str(video_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        mux_failed = result.returncode != 0
    except OSError:
        mux_failed = True

    if mux_failed:
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        shutil.move(str(video_for_mux), str(out_path))
    else:
        pass

    for tmp_path in {silent_path, encoded_path}:
        if tmp_path != out_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return out_path


def make_markdown_summary(tempo, snapped_count1, chosen_track_id, count_motion_df, chunk_df):
    md = "## 検出結果\n"
    md += f"- 推定テンポ: `{tempo}`\n"
    md += f"- 補正後 Count 1: `{snapped_count1:.3f} sec`\n"
    md += f"- 対象 track_id: `{chosen_track_id}`\n"
    md += f"- 解析カウント数: `{len(count_motion_df)}`\n"
    md += f"- 生成チャンク数: `{len(chunk_df)}`\n\n"

    md += "## チャンク一覧\n"
    for _, ch in chunk_df.iterrows():
        md += f"### Chunk {int(ch['chunk_id'])}: Count {ch['count_range']}\n"
        md += f"- 時間: `{ch['start_sec']}`〜`{ch['end_sec']}` sec\n"
        md += f"- カウント数: `{int(ch['num_counts'])}`\n"
        md += f"- 運動負荷: **{ch['motion_load']}**\n"
        md += f"- 主に動く部位: {ch['main_parts']}\n"
        md += f"- 分割理由: {ch['reason']}\n\n"

    md += "---\n"
    md += "このMVPでは、VLM解説はまだ使わず、**カウント境界 + キーポイント移動量**だけでチャンク化しています。\n"
    md += "次に足すなら、各チャンク代表フレームをVLMに渡して「練習ポイント」を短く生成するのがよさそうです。\n"
    return md


# =========================
# Main Gradio process
# =========================

def process_video_for_chunks(
    video_path,
    count1_sec,
    beat_step,
    num_counts,
    target_track_id,
    target_counts_per_chunk,
    min_counts_per_chunk,
    max_counts_per_chunk,
    smooth_window,
):
    if video_path is None:
        return None, None, pd.DataFrame(), pd.DataFrame(), "動画をアップロードしてください。", None, None

    work_dir = tempfile.mkdtemp(prefix="idol_chunk_")
    local_video = str(Path(work_dir) / "input_video.mp4")
    shutil.copy(video_path, local_video)

    info = get_video_info(local_video)

    # 1) Beat / count
    beat_info = detect_beats_from_video(local_video, work_dir, beat_step=int(beat_step))
    count_df, start_beat_idx, snapped_count1 = build_count_table(
        beat_info["beat_times_used"],
        count1_sec=float(count1_sec),
        duration=info["duration"],
        num_counts=int(num_counts),
    )

    # 2) Pose tracking
    pose_csv = str(Path(work_dir) / "pose_keypoints.csv")
    pose_df = run_pose_tracking(local_video, pose_csv, model_name=MODEL_NAME, frame_stride=1, conf=0.25)
    if pose_df.empty:
        return None, None, pd.DataFrame(), pd.DataFrame(), "ボーン検出に失敗しました。全身が見える動画で試してください。", None, pose_csv

    # 3) Track selection
    track_stats = rank_visible_tracks(pose_df)
    if target_track_id is None or int(target_track_id) < 0:
        chosen_track_id = int(track_stats.iloc[0]["track_id"])
    else:
        chosen_track_id = int(target_track_id)
        if chosen_track_id not in set(pose_df["track_id"].unique()):
            chosen_track_id = int(track_stats.iloc[0]["track_id"])

    # 4) Motion intensity
    motion_df = compute_frame_motion(
        pose_df,
        chosen_track_id,
        conf_th=0.3,
        smooth_window=int(smooth_window),
    )
    motion_csv = str(Path(work_dir) / "frame_motion.csv")
    motion_df.to_csv(motion_csv, index=False)

    # 5) Count aggregation
    count_motion_df = aggregate_motion_by_count(count_df, motion_df)
    count_motion_df = attach_choreo_features(count_motion_df, count_df, pose_df, chosen_track_id)
    count_motion_csv = str(Path(work_dir) / "count_motion.csv")
    count_motion_df.to_csv(count_motion_csv, index=False)

    # 6) Choreo-part chunking
    chunk_df = choose_choreo_part_chunks(
        count_motion_df,
        target_counts_per_chunk=int(target_counts_per_chunk),
        min_counts_per_chunk=int(min_counts_per_chunk),
        max_counts_per_chunk=int(max_counts_per_chunk),
        nice_end_counts=(2, 4, 6, 8),
    )
    chunk_csv = str(Path(work_dir) / "chunks.csv")
    chunk_df.to_csv(chunk_csv, index=False)

    # 7) Visualizations
    plot_path = str(Path(work_dir) / "motion_chunks.png")
    plot_motion_counts(count_motion_df, chunk_df, plot_path)

    contact_sheet_path = str(Path(work_dir) / "chunk_contact_sheet.jpg")
    make_chunk_contact_sheet(local_video, pose_df, chosen_track_id, chunk_df, motion_df, contact_sheet_path)

    # 8) Summary
    summary_md = make_markdown_summary(
        tempo=beat_info["tempo"],
        snapped_count1=snapped_count1,
        chosen_track_id=chosen_track_id,
        count_motion_df=count_motion_df,
        chunk_df=chunk_df,
    )

    # 表示用に丸める
    show_count = count_motion_df[[
        "global_count", "count_in_eight", "start_sec", "end_sec",
        "motion_sum", "motion_ratio", "top_parts"
    ]].copy()
    for col in ["start_sec", "end_sec", "motion_sum", "motion_ratio"]:
        show_count[col] = show_count[col].astype(float).round(4)

    show_chunk = chunk_df.copy()

    return (
        plot_path,
        contact_sheet_path,
        show_chunk,
        show_count,
        summary_md,
        chunk_csv,
        pose_csv,
    )

# =========================
# FastAPI Web App
# =========================

import uuid
import json
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = APP_DIR / "frontend"
FIGURE_DIR = APP_DIR / "figure"
WORK_DIR = APP_DIR / "work"
ensure_dir(FIGURE_DIR)
ensure_dir(WORK_DIR)

app = FastAPI(title="IDOL Pin Loop Chunking Web App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NO_CACHE_HEADERS = {"Cache-Control": "no-store, max-age=0"}


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE_HEADERS)
        return response


app.mount("/assets", NoCacheStaticFiles(directory=str(FRONTEND_DIR)), name="assets")
app.mount("/figure", StaticFiles(directory=str(FIGURE_DIR)), name="figure")
app.mount("/work", StaticFiles(directory=str(WORK_DIR)), name="work")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"), headers=NO_CACHE_HEADERS)


def detail_to_params(detail_level: str):
    if detail_level == "細かめ":
        return dict(target_counts_per_chunk=3, min_counts_per_chunk=2, max_counts_per_chunk=6)
    if detail_level == "粗め":
        return dict(target_counts_per_chunk=8, min_counts_per_chunk=4, max_counts_per_chunk=16)
    return dict(target_counts_per_chunk=4, min_counts_per_chunk=2, max_counts_per_chunk=8)


def df_to_records(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    out = df.copy()
    # JSONに落としやすくする
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notnull(out), None)
    return out.to_dict(orient="records")


def choose_count_number(beat_times_used, start_idx, max_counts=None):
    available = max(0, len(beat_times_used) - start_idx - 1)
    if available <= 0:
        return 8
    if max_counts is None:
        return int(max(8, available))
    # 明示的に上限が指定された場合だけ、その範囲に収める
    return int(min(max_counts, max(8, available)))


def maybe_adjust_beat_step(video_path, work_dir):
    """
    UIには出さない内部パラメータ。
    まずbeat_step=1で検出し、ビート間隔が短すぎる場合のみ2にする。
    """
    beat_info = detect_beats_from_video(video_path, work_dir, beat_step=1)
    beats = beat_info["beat_times_used"]
    if len(beats) >= 4:
        median_interval = float(np.median(np.diff(beats)))
        # 0.28秒未満なら細かく拾いすぎの可能性が高い
        if median_interval < 0.28:
            beat_info = detect_beats_from_video(video_path, work_dir, beat_step=2)
            beat_info["auto_beat_step"] = 2
        else:
            beat_info["auto_beat_step"] = 1
    else:
        beat_info["auto_beat_step"] = 1
    return beat_info


def run_full_analysis(session_dir: Path, video_path: Path, detail_level: str):
    info = get_video_info(str(video_path))

    # Count 1はUIに出さず、先頭付近の検出ビートに合わせる
    beat_info = maybe_adjust_beat_step(str(video_path), str(session_dir))
    beat_times = beat_info["beat_times_used"]
    if len(beat_times) < 2:
        raise RuntimeError("ビート検出に失敗しました。音声付きの短い動画で試してください。")
    np.save(session_dir / "beat_times_used.npy", np.asarray(beat_times, dtype=float))

    count1_sec = 0.0
    start_idx = 0
    num_counts = choose_count_number(beat_times, start_idx)

    count_df, start_beat_idx, snapped_count1 = build_count_table(
        beat_times,
        count1_sec=count1_sec,
        duration=info["duration"],
        num_counts=num_counts,
    )

    pose_csv = str(session_dir / "pose_keypoints.csv")
    pose_df = run_pose_tracking(str(video_path), pose_csv, model_name=MODEL_NAME, frame_stride=1, conf=0.25)
    if pose_df.empty:
        raise RuntimeError("ボーン検出に失敗しました。全身が見える動画で試してください。")

    track_stats = rank_visible_tracks(pose_df)
    chosen_track_id = int(track_stats.iloc[0]["track_id"])

    motion_df = compute_frame_motion(
        pose_df,
        chosen_track_id,
        conf_th=0.3,
        smooth_window=5,
    )
    motion_df.to_csv(session_dir / "frame_motion.csv", index=False)

    count_motion_df = aggregate_motion_by_count(count_df, motion_df)
    count_motion_df = attach_choreo_features(count_motion_df, count_df, pose_df, chosen_track_id)
    count_motion_df.to_csv(session_dir / "count_motion.csv", index=False)

    chunk_df = rechunk_from_count_motion(session_dir, detail_level)

    meta = {
        "session_id": session_dir.name,
        "video_filename": video_path.name,
        "video_url": f"/work/{session_dir.name}/{video_path.name}",
        "duration": info["duration"],
        "fps": info["fps"],
        "tempo": beat_info.get("tempo"),
        "auto_beat_step": beat_info.get("auto_beat_step", 1),
        "snapped_count1": snapped_count1,
        "count1_sec": count1_sec,
        "chosen_track_id": chosen_track_id,
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return build_response(session_dir, detail_level)


def rechunk_from_count_motion(session_dir: Path, detail_level: str):
    count_motion_path = session_dir / "count_motion.csv"
    if not count_motion_path.exists():
        raise RuntimeError("count_motion.csvが見つかりません。先に動画を解析してください。")

    count_motion_df = pd.read_csv(count_motion_path)
    params = detail_to_params(detail_level)
    chunk_df = choose_choreo_part_chunks(
        count_motion_df,
        target_counts_per_chunk=params["target_counts_per_chunk"],
        min_counts_per_chunk=params["min_counts_per_chunk"],
        max_counts_per_chunk=params["max_counts_per_chunk"],
        nice_end_counts=(2, 4, 6, 8),
    )
    chunk_df.to_csv(session_dir / "chunks.csv", index=False)
    return chunk_df


def load_or_detect_beat_times(session_dir: Path, meta: dict):
    beat_path = session_dir / "beat_times_used.npy"
    if beat_path.exists():
        return np.load(beat_path)

    video_filename = meta.get("video_filename")
    if not video_filename:
        raise RuntimeError("動画情報が見つかりません。先に動画を解析してください。")

    video_path = session_dir / video_filename
    if not video_path.exists():
        raise RuntimeError("動画ファイルが見つかりません。")

    beat_info = maybe_adjust_beat_step(str(video_path), str(session_dir))
    beat_times = np.asarray(beat_info["beat_times_used"], dtype=float)
    np.save(beat_path, beat_times)
    meta["auto_beat_step"] = beat_info.get("auto_beat_step", meta.get("auto_beat_step", 1))
    meta["tempo"] = beat_info.get("tempo", meta.get("tempo"))
    return beat_times


def reset_count1_for_session(session_dir: Path, count1_sec: float, detail_level: str):
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        raise RuntimeError("meta.jsonが見つかりません。先に動画を解析してください。")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    video_filename = meta.get("video_filename")
    if not video_filename:
        raise RuntimeError("動画情報が見つかりません。")

    video_path = session_dir / video_filename
    motion_path = session_dir / "frame_motion.csv"
    if not video_path.exists() or not motion_path.exists():
        raise RuntimeError("カウント再設定に必要な解析結果が見つかりません。")

    info = get_video_info(str(video_path))
    beat_times = load_or_detect_beat_times(session_dir, meta)
    if len(beat_times) < 2:
        raise RuntimeError("ビート検出に失敗しました。")

    start_idx = find_nearest_index(beat_times, float(count1_sec))
    num_counts = choose_count_number(beat_times, start_idx)
    count_df, start_beat_idx, snapped_count1 = build_count_table(
        beat_times,
        count1_sec=float(count1_sec),
        duration=info["duration"],
        num_counts=num_counts,
    )

    motion_df = pd.read_csv(motion_path)
    count_motion_df = aggregate_motion_by_count(count_df, motion_df)
    pose_path = session_dir / "pose_keypoints.csv"
    if pose_path.exists() and meta.get("chosen_track_id") is not None:
        pose_df = pd.read_csv(pose_path)
        count_motion_df = attach_choreo_features(count_motion_df, count_df, pose_df, int(meta["chosen_track_id"]))
    count_motion_df.to_csv(session_dir / "count_motion.csv", index=False)
    rechunk_from_count_motion(session_dir, detail_level)

    meta["count1_sec"] = float(count1_sec)
    meta["snapped_count1"] = snapped_count1
    meta["start_beat_idx"] = int(start_beat_idx)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_chunk_end_trim(chunk_df):
    out = chunk_df.copy()
    if out.empty or "start_sec" not in out.columns or "end_sec" not in out.columns:
        return out

    if "raw_end_sec" not in out.columns:
        out["raw_end_sec"] = out["end_sec"].astype(float)
        trimmed = out["raw_end_sec"].astype(float) - CHUNK_END_EARLY_SEC
        min_end = out["start_sec"].astype(float) + 0.05
        out["end_sec"] = np.maximum(min_end, trimmed)

    return out


def build_response(session_dir: Path, detail_level: str):
    meta_path = session_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    count_motion_df = pd.read_csv(session_dir / "count_motion.csv")
    chunk_df = apply_chunk_end_trim(pd.read_csv(session_dir / "chunks.csv"))

    # 表示に必要な列だけにする
    count_cols = [
        "global_count", "count_in_eight", "start_sec", "end_sec",
        "motion_sum", "motion_ratio", "top_parts"
    ]
    count_out = count_motion_df[count_cols].copy()
    for col in ["start_sec", "end_sec", "motion_sum", "motion_ratio"]:
        count_out[col] = count_out[col].astype(float).round(5)

    chunk_out = chunk_df.copy()
    for col in ["start_sec", "end_sec", "motion_sum", "load_ratio"]:
        if col in chunk_out.columns:
            chunk_out[col] = chunk_out[col].astype(float).round(5)

    def versioned_work_url(url):
        if not url:
            return url
        filename = str(url).split("/")[-1].split("?")[0]
        path = session_dir / filename
        if path.exists():
            return f"{url}?v={int(path.stat().st_mtime)}"
        return url

    return {
        "ok": True,
        "session_id": session_dir.name,
        "detail_level": detail_level,
        "video_url": versioned_work_url(meta.get("video_url")),
        "duration": meta.get("duration"),
        "tempo": meta.get("tempo"),
        "auto_beat_step": meta.get("auto_beat_step"),
        "chunks": df_to_records(chunk_out),
        "counts": df_to_records(count_out),
    }


@app.post("/api/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    detail_level: str = Form("普通"),
):
    session_id = uuid.uuid4().hex[:12]
    session_dir = WORK_DIR / session_id
    ensure_dir(session_dir)

    suffix = Path(video.filename or "input.mp4").suffix.lower()
    if suffix not in [".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"]:
        suffix = ".mp4"

    video_path = session_dir / f"input{suffix}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        result = run_full_analysis(session_dir, video_path, detail_level)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500,
        )


@app.post("/api/rechunk/{session_id}")
async def rechunk(session_id: str, detail_level: str = Form("普通")):
    session_dir = WORK_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session not found")

    try:
        rechunk_from_count_motion(session_dir, detail_level)
        return JSONResponse(build_response(session_dir, detail_level))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500,
        )


@app.post("/api/count1/{session_id}")
async def set_count1(
    session_id: str,
    count1_sec: float = Form(...),
    detail_level: str = Form("普通"),
):
    session_dir = WORK_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session not found")

    try:
        reset_count1_for_session(session_dir, count1_sec, detail_level)
        return JSONResponse(build_response(session_dir, detail_level))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

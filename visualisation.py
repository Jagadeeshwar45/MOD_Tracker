"""
Optional Enhancements for MOT Tracker
======================================
Run after tracker.py has produced outputs/tracks.json and the annotated video.

All output files are saved to outputs/visualisation/

Covers:
  1. Trajectory visualization
  2. Movement heatmap
  3. Bird's-eye / top-view projection
  4. Object count over time
  5. Team / role clustering (jersey color k-means)
  6. Speed estimation + movement statistics
  7. Simple evaluation metrics (ID switches, fragmentation, coverage)
  8. Model comparison (yolov8n vs yolov8s vs yolov8m on a sample clip)
"""

import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    tracks_json      = "outputs/tracks.json"
    input_video      = "input_video.mp4"
    annotated_video  = "outputs/output_annotated.mp4"
    output_dir       = "outputs/visualisation"

    fps              = 25.0          # set to your video's actual FPS
    pixels_per_meter = 8.0           # calibrate from a known pitch dimension
                                     # cricket pitch = 20.12m, measure in pixels

    # Bird's-eye: 4 corner points of the pitch in the video frame (x, y)
    # Click on: [top-left, top-right, bottom-right, bottom-left] of the pitch
    # Set these by inspecting a frame — see helper below
    pitch_src_points = None          # e.g. np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    pitch_dst_size   = (400, 800)    # (width, height) of the top-down output canvas

    # Team clustering
    n_teams          = 3             # typically 2 teams + 1 umpire role
    cluster_sample_frames = 50       # frames to sample for jersey color extraction

    # Model comparison
    compare_models = ["yolov8n", "yolov8s", "yolov8m"]
    compare_n_frames = 50           # number of frames to benchmark
    compare_conf     = 0.25
# ══════════════════════════════════════════════════════════════════════════════


Path(Config.output_dir).mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_tracks(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # Keys come back as strings from JSON — convert to int
    return {int(k): v for k, v in raw.items()}


def get_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or Config.fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return W, H, fps, total


def build_history(tracks: dict) -> dict:
    """
    Returns dict: track_id -> list of (frame_idx, cx, cy, bbox)
    sorted by frame index.
    """
    history = defaultdict(list)
    for frame_idx, frame_data in sorted(tracks.items()):
        for obj in frame_data:
            tid = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            history[tid].append((frame_idx, cx, cy, obj["bbox"]))
    return dict(history)


def save(fig, name: str):
    path = f"{Config.output_dir}/{name}"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. TRAJECTORY VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_trajectories(tracks: dict, W: int, H: int, top_n: int = 20):
    """
    Colored trajectory lines for the top N most-tracked IDs.
    Line opacity fades from transparent (start) to solid (end).
    """
    print("\n[1] Generating trajectory visualization...")
    history = build_history(tracks)

    top_ids = sorted(history, key=lambda k: len(history[k]), reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_facecolor("#0a1628")
    fig.patch.set_facecolor("#0a1628")
    ax.set_title(f"Player trajectories — top {top_n} tracks", color="white", fontsize=11, pad=8)
    ax.axis("off")

    cmap = plt.cm.get_cmap("tab20", len(top_ids))

    for i, tid in enumerate(top_ids):
        pts = np.array([(cx, cy) for _, cx, cy, _ in history[tid]])
        if len(pts) < 2:
            continue
        segments = [pts[j:j+2] for j in range(len(pts) - 1)]
        alphas   = np.linspace(0.05, 1.0, len(segments))
        color    = cmap(i)
        lc = LineCollection(
            segments,
            colors=[(*color[:3], a) for a in alphas],
            linewidths=1.5
        )
        ax.add_collection(lc)
        # Label at final position
        ax.text(pts[-1, 0] + 4, pts[-1, 1], f"ID:{tid}",
                color=color, fontsize=6, va="center")

    save(fig, "trajectories.png")


# ══════════════════════════════════════════════════════════════════════════════
#  2. MOVEMENT HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def generate_heatmap(tracks: dict, W: int, H: int):
    """
    Gaussian-blurred heatmap of all tracked center positions.
    Brighter = more time spent at that location.
    """
    print("\n[2] Generating movement heatmap...")
    heat = np.zeros((H, W), dtype=np.float32)

    for frame_data in tracks.values():
        for obj in frame_data:
            x1, y1, x2, y2 = obj["bbox"]
            cx = int(np.clip((x1 + x2) / 2, 0, W - 1))
            cy = int(np.clip((y1 + y2) / 2, 0, H - 1))
            heat[cy, cx] += 1.0

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=30, sigmaY=30)
    if heat.max() > 0:
        heat = (heat / heat.max() * 255).astype(np.uint8)

    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    # Overlay a semi-transparent field background
    field_bg = np.zeros((H, W, 3), dtype=np.uint8)
    field_bg[:] = (20, 80, 20)   # dark green
    blended = cv2.addWeighted(field_bg, 0.3, colored, 0.7, 0)

    out_path = f"{Config.output_dir}/heatmap.png"
    cv2.imwrite(out_path, blended)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. BIRD'S-EYE / TOP-VIEW PROJECTION
# ══════════════════════════════════════════════════════════════════════════════

def pick_pitch_corners(video_path: str):
    """
    Interactive helper — opens the first frame and lets you click 4 pitch corners.
    Click order: top-left → top-right → bottom-right → bottom-left
    Press 'q' after clicking all 4 points.

    Returns np.float32 array of shape (4, 2).
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("  Could not read video for corner picking.")
        return None

    points = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(frame, str(len(points)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Click 4 pitch corners (TL, TR, BR, BL) — press q when done", frame)

    cv2.imshow("Click 4 pitch corners (TL, TR, BR, BL) — press q when done", frame)
    cv2.setMouseCallback(
        "Click 4 pitch corners (TL, TR, BR, BL) — press q when done", click
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 4:
        arr = np.float32(points)
        print(f"  Corners picked: {arr.tolist()}")
        return arr
    return None


def generate_birdseye(tracks: dict, src_points: np.ndarray = None,
                      dst_size: tuple = (400, 800), video_path: str = None):
    """
    Projects all player center positions onto a top-down pitch view.

    src_points: 4 corner points of the pitch in the original frame
                (top-left, top-right, bottom-right, bottom-left)
    dst_size:   (width, height) of the output canvas
    """
    print("\n[3] Generating bird's-eye projection...")

    if src_points is None:
        print("  pitch_src_points not set in Config.")
        if video_path:
            print("  Launching interactive corner picker...")
            src_points = pick_pitch_corners(video_path)
        if src_points is None:
            print("  Skipping bird's-eye — no pitch corners defined.")
            print("  Tip: set Config.pitch_src_points manually after running pick_pitch_corners()")
            return

    dW, dH = dst_size
    dst_points = np.float32([
        [0,  0 ],    # top-left
        [dW, 0 ],    # top-right
        [dW, dH],    # bottom-right
        [0,  dH],    # bottom-left
    ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Build canvas — cricket pitch green
    canvas = np.ones((dH, dW, 3), dtype=np.uint8) * np.array([34, 139, 34], dtype=np.uint8)

    # Draw pitch strip (center rectangle)
    pw = int(dW * 0.18)
    px = (dW - pw) // 2
    cv2.rectangle(canvas, (px, int(dH * 0.1)), (px + pw, int(dH * 0.9)),
                  (180, 160, 100), -1)
    # Crease lines
    for y_frac in [0.15, 0.85]:
        y = int(dH * y_frac)
        cv2.line(canvas, (px, y), (px + pw, y), (255, 255, 255), 2)

    history = build_history(tracks)
    cmap    = plt.cm.get_cmap("tab20", min(len(history), 20))

    # Project each track's path onto the top-down view
    for i, (tid, pts_raw) in enumerate(list(history.items())[:20]):
        color_rgb = cmap(i % 20)[:3]
        color_bgr = (int(color_rgb[2]*255), int(color_rgb[1]*255), int(color_rgb[0]*255))

        proj_pts = []
        for _, cx, cy, _ in pts_raw:
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(pt, M)[0][0]
            wx, wy = int(np.clip(warped[0], 0, dW-1)), int(np.clip(warped[1], 0, dH-1))
            proj_pts.append((wx, wy))

        # Draw trail
        for j in range(1, len(proj_pts)):
            cv2.line(canvas, proj_pts[j-1], proj_pts[j], color_bgr, 2, cv2.LINE_AA)

        # Draw final position dot
        if proj_pts:
            cv2.circle(canvas, proj_pts[-1], 5, color_bgr, -1)
            cv2.putText(canvas, str(tid), (proj_pts[-1][0]+6, proj_pts[-1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)

    out_path = f"{Config.output_dir}/birdseye.png"
    cv2.imwrite(out_path, canvas)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  4. OBJECT COUNT OVER TIME
# ══════════════════════════════════════════════════════════════════════════════

def plot_count_over_time(tracks: dict, fps: float):
    """
    Line chart of active track count per frame, with a rolling average overlay.
    """
    print("\n[4] Generating object count over time plot...")

    frame_indices = sorted(tracks.keys())
    counts  = [len(tracks[fi]) for fi in frame_indices]
    times   = [fi / fps for fi in frame_indices]

    # Rolling average (window = 1 second of frames)
    window  = max(1, int(fps))
    rolling = np.convolve(counts, np.ones(window)/window, mode="same")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(times, counts, alpha=0.15, color="#378ADD")
    ax.plot(times, counts,   color="#378ADD", linewidth=0.8, alpha=0.7, label="Per frame")
    ax.plot(times, rolling,  color="#D85A30", linewidth=2.0, label=f"{window}-frame rolling avg")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Active tracks", fontsize=11)
    ax.set_title("Active object count over time", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    save(fig, "count_over_time.png")


# ══════════════════════════════════════════════════════════════════════════════
#  5. TEAM / ROLE CLUSTERING (jersey color k-means)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_teams(tracks: dict, video_path: str, n_clusters: int = 3,
                  sample_frames: int = 50):
    """
    Extracts the dominant jersey color per track ID by:
      1. Sampling up to `sample_frames` frames from the video.
      2. Cropping each player's bounding box (top 60% = jersey area).
      3. Computing mean HSV color of the crop.
      4. K-means clustering all track colors into n_clusters groups.

    Saves a scatter plot and returns dict: track_id -> cluster_label.
    """
    print("\n[5] Running team/role clustering...")

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick evenly spaced sample frames
    sample_idxs = set(
        int(total * i / sample_frames) for i in range(sample_frames)
    )

    track_colors: dict = defaultdict(list)   # tid -> list of mean HSV colors
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx not in sample_idxs:
            continue
        if frame_idx not in tracks:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for obj in tracks[frame_idx]:
            tid        = obj["id"]
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            h          = y2 - y1
            # Use top 60% of box as jersey (avoids legs/feet)
            jersey_y2  = y1 + int(h * 0.60)
            crop       = hsv[y1:jersey_y2, x1:x2]
            if crop.size == 0:
                continue
            mean_color = crop.mean(axis=(0, 1))   # mean H, S, V
            track_colors[tid].append(mean_color)

    cap.release()

    if len(track_colors) < n_clusters:
        print(f"  Not enough tracks ({len(track_colors)}) for {n_clusters} clusters. Skipping.")
        return {}

    # Average color per track
    tids        = list(track_colors.keys())
    avg_colors  = np.array([np.mean(track_colors[t], axis=0) for t in tids])

    # K-means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(avg_colors)

    team_map = {tid: int(label) for tid, label in zip(tids, labels)}

    # ── Plot ──────────────────────────────────────────────────────────────────
    cluster_names  = [f"Team A", "Team B", "Umpire/Other"][:n_clusters]
    cluster_colors = ["#378ADD", "#E24B4A", "#F0997B"][:n_clusters]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter of H vs S colored by cluster
    for cl in range(n_clusters):
        mask = labels == cl
        axes[0].scatter(
            avg_colors[mask, 0],   # Hue
            avg_colors[mask, 1],   # Saturation
            c=cluster_colors[cl],
            label=cluster_names[cl],
            s=60, alpha=0.8, edgecolors="white", linewidths=0.5
        )
    axes[0].set_xlabel("Hue (HSV)", fontsize=11)
    axes[0].set_ylabel("Saturation (HSV)", fontsize=11)
    axes[0].set_title("Jersey color clusters (Hue vs Saturation)", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)

    # Right: bar chart of cluster sizes
    counts = [np.sum(labels == cl) for cl in range(n_clusters)]
    bars   = axes[1].bar(cluster_names[:n_clusters], counts,
                         color=cluster_colors[:n_clusters], edgecolor="white",
                         linewidth=0.5)
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, str(count),
                     ha="center", va="bottom", fontsize=11)
    axes[1].set_ylabel("Number of tracks", fontsize=11)
    axes[1].set_title("Track count per cluster", fontsize=11)
    axes[1].grid(True, alpha=0.2, axis="y")

    fig.suptitle("Team / role clustering via jersey color k-means", fontsize=13)
    fig.tight_layout()
    save(fig, "team_clustering.png")

    # Save cluster assignments
    out_path = f"{Config.output_dir}/team_assignments.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in team_map.items()}, f, indent=2)
    print(f"  Saved: {out_path}")

    return team_map


# ══════════════════════════════════════════════════════════════════════════════
#  6. SPEED ESTIMATION + MOVEMENT STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def estimate_speeds(tracks: dict, fps: float, pixels_per_meter: float):
    """
    Estimates per-track speed in m/s using frame-to-frame center displacement.
    Computes: median speed, max speed, total distance traveled.

    Saves a bar chart of top 15 tracks by median speed.
    """
    print("\n[6] Estimating speeds and movement statistics...")

    history = build_history(tracks)
    stats   = {}

    for tid, pts in history.items():
        if len(pts) < 2:
            continue
        speeds, distances = [], []
        for j in range(1, len(pts)):
            dt = (pts[j][0] - pts[j-1][0]) / fps
            if dt <= 0:
                continue
            dx = pts[j][1] - pts[j-1][1]
            dy = pts[j][2] - pts[j-1][2]
            dist_px = np.sqrt(dx**2 + dy**2)
            dist_m  = dist_px / pixels_per_meter
            distances.append(dist_m)
            speeds.append(dist_m / dt)

        if not speeds:
            continue

        speeds_arr = np.array(speeds)
        # Remove outliers (> 15 m/s is physically impossible for a fielder)
        speeds_arr = speeds_arr[speeds_arr < 15.0]
        if len(speeds_arr) == 0:
            continue

        stats[tid] = {
            "median_speed_ms":  float(np.median(speeds_arr)),
            "max_speed_ms":     float(np.max(speeds_arr)),
            "total_distance_m": float(np.sum(distances)),
            "track_length_frames": len(pts),
        }

    if not stats:
        print("  No valid speed data.")
        return stats

    # Print summary
    print(f"  Tracks with speed data: {len(stats)}")
    top5 = sorted(stats.items(), key=lambda x: -x[1]["median_speed_ms"])[:5]
    print("  Top 5 by median speed:")
    for tid, s in top5:
        print(f"    ID {tid:4d} — median {s['median_speed_ms']:.2f} m/s  "
              f"max {s['max_speed_ms']:.2f} m/s  "
              f"dist {s['total_distance_m']:.1f} m")

    # ── Plot ──────────────────────────────────────────────────────────────────
    top15 = sorted(stats.items(), key=lambda x: -x[1]["median_speed_ms"])[:15]
    tids_plot   = [str(t[0]) for t in top15]
    med_speeds  = [t[1]["median_speed_ms"] for t in top15]
    max_speeds  = [t[1]["max_speed_ms"]    for t in top15]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: median vs max speed
    x = np.arange(len(tids_plot))
    w = 0.38
    axes[0].bar(x - w/2, med_speeds, w, label="Median speed", color="#378ADD", alpha=0.85)
    axes[0].bar(x + w/2, max_speeds, w, label="Max speed",    color="#D85A30", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"ID:{t}" for t in tids_plot], rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("Speed (m/s)", fontsize=11)
    axes[0].set_title("Top 15 tracks — median vs max speed", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2, axis="y")

    # Right: total distance traveled
    top15_dist = sorted(stats.items(), key=lambda x: -x[1]["total_distance_m"])[:15]
    axes[1].barh(
        [f"ID:{t[0]}" for t in top15_dist],
        [t[1]["total_distance_m"] for t in top15_dist],
        color="#1D9E75", alpha=0.85
    )
    axes[1].set_xlabel("Total distance (m)", fontsize=11)
    axes[1].set_title("Top 15 tracks — total distance covered", fontsize=11)
    axes[1].grid(True, alpha=0.2, axis="x")
    axes[1].invert_yaxis()

    fig.suptitle("Speed estimation and movement statistics", fontsize=13)
    fig.tight_layout()
    save(fig, "speed_stats.png")

    # Save full stats JSON
    out_path = f"{Config.output_dir}/speed_stats.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)
    print(f"  Saved: {out_path}")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  7. EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_evaluation_metrics(tracks: dict, fps: float, total_frames: int):
    """
    Computes self-consistency metrics from the tracking output:

    - Total unique IDs          : total distinct track IDs assigned
    - ID switches (proxy)       : IDs that appear, disappear, then reappear
                                  (gap > 1s = likely a re-ID failure)
    - Track fragmentation rate  : % of tracks that are fragmented
    - Avg track duration        : mean number of seconds a track is alive
    - Detection coverage        : % of processed frames with >= 1 detection
    - Avg detections per frame  : mean active tracks per frame
    - Short tracks (<1s)        : tracks too brief to be meaningful

    Saves a metrics summary PNG and prints results.
    """
    print("\n[7] Computing evaluation metrics...")

    history      = build_history(tracks)
    processed    = len(tracks)
    gap_threshold = fps * 1.0   # 1 second gap = likely fragmentation

    fragmented_ids   = []
    track_durations  = []
    id_switch_count  = 0
    short_tracks     = 0

    for tid, pts in history.items():
        frame_indices = [p[0] for p in pts]
        duration_frames = frame_indices[-1] - frame_indices[0] + 1
        duration_sec    = duration_frames / fps
        track_durations.append(duration_sec)

        if duration_sec < 1.0:
            short_tracks += 1

        # Check for gaps in the track (fragmentation proxy)
        gaps = [frame_indices[j] - frame_indices[j-1]
                for j in range(1, len(frame_indices))]
        if any(g > gap_threshold for g in gaps):
            fragmented_ids.append(tid)
            id_switch_count += sum(1 for g in gaps if g > gap_threshold)

    n_tracks         = len(history)
    coverage         = processed / total_frames * 100 if total_frames > 0 else 0
    frames_with_dets = sum(1 for v in tracks.values() if len(v) > 0)
    det_coverage     = frames_with_dets / processed * 100 if processed > 0 else 0
    avg_per_frame    = np.mean([len(v) for v in tracks.values()])
    frag_rate        = len(fragmented_ids) / n_tracks * 100 if n_tracks > 0 else 0
    avg_duration     = np.mean(track_durations) if track_durations else 0

    metrics = {
        "total_unique_ids":         n_tracks,
        "frames_processed":         processed,
        "total_frames":             total_frames,
        "frame_coverage_pct":       round(coverage, 1),
        "detection_coverage_pct":   round(det_coverage, 1),
        "avg_detections_per_frame": round(float(avg_per_frame), 2),
        "id_switches_proxy":        id_switch_count,
        "fragmented_tracks":        len(fragmented_ids),
        "fragmentation_rate_pct":   round(frag_rate, 1),
        "avg_track_duration_sec":   round(float(avg_duration), 2),
        "short_tracks_under_1s":    short_tracks,
    }

    # Print
    print(f"  Total unique IDs          : {metrics['total_unique_ids']}")
    print(f"  Frames processed          : {metrics['frames_processed']} / {total_frames} ({coverage:.1f}%)")
    print(f"  Detection coverage        : {det_coverage:.1f}% of frames have ≥1 detection")
    print(f"  Avg detections/frame      : {avg_per_frame:.2f}")
    print(f"  ID switches (proxy)       : {id_switch_count}")
    print(f"  Fragmented tracks         : {len(fragmented_ids)} ({frag_rate:.1f}%)")
    print(f"  Avg track duration        : {avg_duration:.2f}s")
    print(f"  Short tracks (<1s)        : {short_tracks}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: metric summary bar
    labels_m = ["Unique IDs", "ID switches", "Fragmented\ntracks", "Short\ntracks (<1s)"]
    values_m = [
        metrics["total_unique_ids"],
        metrics["id_switches_proxy"],
        metrics["fragmented_tracks"],
        metrics["short_tracks_under_1s"],
    ]
    colors_m = ["#378ADD", "#E24B4A", "#EF9F27", "#888780"]
    bars = axes[0].bar(labels_m, values_m, color=colors_m, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values_m):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3, str(val),
                     ha="center", va="bottom", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Tracking quality metrics", fontsize=11)
    axes[0].grid(True, alpha=0.2, axis="y")

    # Right: track duration histogram
    axes[1].hist(track_durations, bins=30, color="#378ADD", alpha=0.8, edgecolor="white")
    axes[1].axvline(avg_duration, color="#E24B4A", linewidth=2,
                    linestyle="--", label=f"Mean: {avg_duration:.1f}s")
    axes[1].set_xlabel("Track duration (seconds)", fontsize=11)
    axes[1].set_ylabel("Number of tracks", fontsize=11)
    axes[1].set_title("Track duration distribution", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle("Evaluation metrics — ID stability and track quality", fontsize=13)
    fig.tight_layout()
    save(fig, "evaluation_metrics.png")

    # Save JSON
    out_path = f"{Config.output_dir}/evaluation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {out_path}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  8. MODEL COMPARISON (yolov8n vs yolov8s vs yolov8m)
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(video_path: str, models: list, n_frames: int, conf: float):
    """
    Benchmarks two YOLOv8 models on n_frames evenly sampled from the video.
    Compares: inference speed, detection count, confidence distribution.

    Saves a side-by-side comparison plot.
    """
    print("\n[8] Running model comparison...")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed — skipping model comparison.")
        return

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_idxs = sorted(set(
        int(total * i / n_frames) for i in range(n_frames)
    ))

    # Read sample frames
    frames, fi = [], 0
    while len(frames) < len(sample_idxs):
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        if fi in sample_idxs:
            frames.append(frame)
    cap.release()

    results_per_model = {}

    for model_name in models:
        print(f"  Benchmarking {model_name}...")
        model = YOLO(f"{model_name}.pt")

        det_counts, confidences, times = [], [], []

        for frame in frames:
            t0 = time.perf_counter()
            res = model(frame, conf=conf, classes=[0], verbose=False)[0]
            elapsed = time.perf_counter() - t0

            times.append(elapsed * 1000)   # ms
            det_counts.append(len(res.boxes))
            if len(res.boxes) > 0:
                confidences.extend(res.boxes.conf.cpu().numpy().tolist())

        results_per_model[model_name] = {
            "avg_time_ms":    float(np.mean(times)),
            "avg_detections": float(np.mean(det_counts)),
            "confidences":    confidences,
            "det_counts":     det_counts,
        }
        print(f"    Avg time: {np.mean(times):.1f}ms | "
              f"Avg dets: {np.mean(det_counts):.1f} | "
              f"Frames: {len(frames)}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = ["#378ADD", "#D85A30", "#1D9E75"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    model_names = list(results_per_model.keys())

    # Left: inference speed
    avg_times = [results_per_model[m]["avg_time_ms"] for m in model_names]
    bars = axes[0].bar(model_names, avg_times, color=colors[:len(model_names)],
                       edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, avg_times):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, f"{val:.1f}ms",
                     ha="center", va="bottom", fontsize=11)
    axes[0].set_ylabel("Avg inference time (ms)", fontsize=11)
    axes[0].set_title("Inference speed", fontsize=11)
    axes[0].grid(True, alpha=0.2, axis="y")

    # Middle: avg detection count
    avg_dets = [results_per_model[m]["avg_detections"] for m in model_names]
    bars2 = axes[1].bar(model_names, avg_dets, color=colors[:len(model_names)],
                        edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, avg_dets):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05, f"{val:.1f}",
                     ha="center", va="bottom", fontsize=11)
    axes[1].set_ylabel("Avg detections per frame", fontsize=11)
    axes[1].set_title("Detection count", fontsize=11)
    axes[1].grid(True, alpha=0.2, axis="y")

    # Right: confidence distribution
    for i, m in enumerate(model_names):
        confs = results_per_model[m]["confidences"]
        if confs:
            axes[2].hist(confs, bins=20, alpha=0.6, color=colors[i],
                         label=m, edgecolor="white", linewidth=0.3)
    axes[2].set_xlabel("Detection confidence", fontsize=11)
    axes[2].set_ylabel("Frequency", fontsize=11)
    axes[2].set_title("Confidence distribution", fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.2)

    fig.suptitle(f"Model comparison on {len(frames)}-frame sample (conf={conf})", fontsize=13)
    fig.tight_layout()
    save(fig, "model_comparison.png")

    # Save JSON
    out_path = f"{Config.output_dir}/model_comparison.json"
    summary = {
        m: {k: v for k, v in d.items() if k != "confidences"}
        for m, d in results_per_model.items()
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = Config()

    print("Loading tracks and video metadata...")
    tracks = load_tracks(cfg.tracks_json)
    W, H, fps, total_frames = get_video_meta(cfg.input_video)
    print(f"  Video: {W}x{H} @ {fps:.1f}fps — {total_frames} total frames")
    print(f"  Tracks loaded: {len(tracks)} processed frames")

    # 1. Trajectories
    generate_trajectories(tracks, W, H, top_n=20)

    # 2. Heatmap
    generate_heatmap(tracks, W, H)

    # 3. Bird's-eye projection
    #    If pitch corners not set in Config, launches interactive picker.
    #    To skip entirely, comment out the line below.
    generate_birdseye(
        tracks,
        src_points=cfg.pitch_src_points,
        dst_size=cfg.pitch_dst_size,
        video_path=cfg.input_video,
    )

    # 4. Count over time
    plot_count_over_time(tracks, fps)

    # 5. Team clustering
    team_map = cluster_teams(
        tracks,
        video_path=cfg.input_video,
        n_clusters=cfg.n_teams,
        sample_frames=cfg.cluster_sample_frames,
    )

    # 6. Speed estimation
    speed_stats = estimate_speeds(tracks, fps, cfg.pixels_per_meter)

    # 7. Evaluation metrics
    metrics = compute_evaluation_metrics(tracks, fps, total_frames)

    # 8. Model comparison
    compare_models(
        video_path=cfg.input_video,
        models=cfg.compare_models,
        n_frames=cfg.compare_n_frames,
        conf=cfg.compare_conf,
    )
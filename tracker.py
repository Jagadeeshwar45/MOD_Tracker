"""
Multi-Object Detection and Persistent ID Tracking
==================================================
Uses YOLOv8 for detection and ByteTrack (via supervision) for tracking.

Run:
    python tracker.py

All parameters are configured in the CONFIG block below — no CLI arguments needed.
"""

import json
import os
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG 
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    input_path   = "input_video.mp4"          # path to input video
    output_path  = "outputs/output_annotated.mp4"  # where to save annotated video
    model        = "yolov8s"                  # yolov8n / yolov8s / yolov8m / yolov8l
    conf         = 0.20                       # detection confidence threshold (lower = catches distant players)
    iou          = 0.45                       # NMS IoU threshold
    frame_skip   = 2                          # process every 4th frame — halves total time
    classes      = [0]                        # COCO class IDs: 0=person, 32=sports ball
    save_json    = True                       # save per-frame track data to JSON
    trail_length = 40                         # number of frames to keep movement trails
    use_tiling   = True                       # tile frames for better small-player detection
    min_box_area_ratio  = 0.001              # minimum box area as fraction of frame (removes crowd heads)
    min_box_height_ratio = 0.04              # minimum box height as fraction of frame height
# ══════════════════════════════════════════════════════════════════════════════


# ─── Annotator ────────────────────────────────────────────────────────────────

def id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Returns a stable, deterministic BGR color for a given track ID."""
    np.random.seed(track_id * 37 + 13)
    color = tuple(int(c) for c in np.random.randint(80, 230, 3))
    return color


class TrackAnnotator:
    """Handles all visual annotation: trails, bounding boxes, ID labels, HUD."""

    def __init__(self, trail_length: int = 40):
        self.trail_length = trail_length
        self.trail_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    def update_trails(self, track_ids: np.ndarray, boxes_xyxy: np.ndarray):
        """Store center points per track ID for trail drawing."""
        current_ids = set(int(t) for t in track_ids)
        # Remove stale trails for IDs no longer active
        for tid in list(self.trail_history.keys()):
            if tid not in current_ids:
                del self.trail_history[tid]

        for tid, box in zip(track_ids, boxes_xyxy):
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            self.trail_history[int(tid)].append((cx, cy))
            if len(self.trail_history[int(tid)]) > self.trail_length:
                self.trail_history[int(tid)].pop(0)

    def draw_frame(
        self,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,
        track_ids: np.ndarray,
        confidences: np.ndarray = None,
        frame_count: int = 0,
        total_unique: int = 0,
    ) -> np.ndarray:
        """
        Draw trails, bounding boxes, ID labels, and a HUD status bar.

        Args:
            frame:        BGR frame
            boxes_xyxy:   (N, 4) array of [x1, y1, x2, y2]
            track_ids:    (N,) integer track IDs
            confidences:  (N,) detection confidence scores
            frame_count:  current frame index
            total_unique: total distinct IDs seen across the video

        Returns:
            Annotated BGR frame (copy of input).
        """
        out = frame.copy()

        if len(track_ids) > 0:
            self.update_trails(track_ids, boxes_xyxy)

        # ── Draw trails (rendered under boxes) ──────────────────────────────
        for tid, pts in self.trail_history.items():
            color = id_to_color(tid)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(2 * alpha))
                cv2.line(out, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # ── Draw boxes and ID labels ─────────────────────────────────────────
        for i, (box, tid) in enumerate(zip(boxes_xyxy, track_ids)):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            tid = int(tid)
            color = id_to_color(tid)

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label text
            label = f"ID:{tid}"
            if confidences is not None and len(confidences) > i:
                label += f" {confidences[i]:.2f}"

            # Label background pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx1 = x1
            ly1 = max(y1 - th - 6, 0)
            lx2 = x1 + tw + 6
            ly2 = y1
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(
                out, label, (lx1 + 3, ly2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

        # ── HUD bar at bottom ────────────────────────────────────────────────
        h, w = out.shape[:2]
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - 32), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        hud = (
            f"Frame: {frame_count}  |  "
            f"Active: {len(track_ids)}  |  "
            f"Total seen: {total_unique}"
        )
        cv2.putText(
            out, hud, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA
        )

        return out


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class MOTPipeline:
    """
    End-to-end Multi-Object Tracking pipeline.

    Flow:
      1. YOLOv8 detects persons using tiled inference (catches distant players).
      2. Dynamic field-boundary estimation removes audience detections.
      3. Box size filter removes small crowd heads that pass the ROI check.
      4. ByteTrack assigns and propagates persistent IDs across frames.
      5. TrackAnnotator draws trails, boxes, IDs, and a HUD overlay.
      6. Annotated frames are written to an output MP4.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/screenshots").mkdir(exist_ok=True)

        print(f"Loading model: {cfg.model}.pt")
        self.model = YOLO(f"{cfg.model}.pt")   # auto-downloads on first run

        # ByteTrack tracker
        self.tracker = sv.ByteTrack(
            track_activation_threshold=cfg.conf,
            lost_track_buffer=60,          # keep lost tracks alive for 2s @ 30fps
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        self.annotator = TrackAnnotator(trail_length=cfg.trail_length)
        self.track_log: dict = {}
        self.all_ids: set = set()

    # ── Video helpers ─────────────────────────────────────────────────────────

    def _open_video(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, fps, width, height, total

    def _make_writer(self, path: str, fps: float, w: int, h: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, fps, (w, h))

    # ── Dynamic ROI ───────────────────────────────────────────────────────────

    def _estimate_field_top(self, frame: np.ndarray) -> float:
        """
        Estimates what vertical fraction of the frame is non-field (stands/sky)
        by finding where continuous green (grass) begins.

        Returns a float 0.0–0.70 — detections whose center-y is above this
        fraction of the frame height are suppressed.
        """
        H = frame.shape[0]
        # Green channel dominance: G > R and G > B by a margin
        green_mask = (
            (frame[:, :, 1].astype(int) - frame[:, :, 2].astype(int) > 20) &
            (frame[:, :, 1].astype(int) - frame[:, :, 0].astype(int) > 10)
        )
        # Scan rows top-to-bottom; first row with >15% green pixels = field start
        for row in range(H):
            if green_mask[row].mean() > 0.15:
                return max(0.0, (row / H) - 0.05)   # small buffer above grass line
        return 0.35   # fallback if no clear grass found

    # ── Detection (tiled) ─────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Tiled YOLOv8 detection:
          - Splits the frame into 5 overlapping tiles (full + 4 quadrants).
          - Translates crop coordinates back to full-frame space.
          - Merges with NMS to remove duplicates from overlapping tiles.
          - Applies dynamic ROI + box size filters to suppress audience.

        Returns a sv.Detections object with only valid player detections.
        """
        H, W = frame.shape[:2]
        all_boxes, all_confs, all_class_ids = [], [], []

        if self.cfg.use_tiling:
            # 3-tile horizontal split — covers wide broadcast shots efficiently
            ox = int(W * 0.10)
            tiles = [
                (0,           0, W,             H),   # full frame (catches mid-range)
                (0,           0, W//2 + ox,     H),   # left half  (catches left-side players)
                (W//2 - ox,   0, W,             H),   # right half (catches right-side players)
            ]
        else:
            tiles = [(0, 0, W, H)]

        for (tx1, ty1, tx2, ty2) in tiles:
            crop = frame[ty1:ty2, tx1:tx2]
            results = self.model(
                crop,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                classes=self.cfg.classes,
                verbose=False,
                imgsz=640,
            )[0]
            dets = sv.Detections.from_ultralytics(results)
            if len(dets) == 0:
                continue
            # Translate tile-local coords back to full-frame coords
            dets.xyxy[:, 0] += tx1
            dets.xyxy[:, 1] += ty1
            dets.xyxy[:, 2] += tx1
            dets.xyxy[:, 3] += ty1
            all_boxes.append(dets.xyxy)
            all_confs.append(dets.confidence)
            all_class_ids.append(dets.class_id)

        if not all_boxes:
            return sv.Detections.empty()

        # Merge all tile results
        merged = sv.Detections(
            xyxy=np.vstack(all_boxes),
            confidence=np.concatenate(all_confs),
            class_id=np.concatenate(all_class_ids),
        )

        # NMS to remove duplicates from overlapping tiles
        merged = merged.with_nms(threshold=0.45)

        if len(merged) == 0:
            return merged

        # ── Filter 1: Dynamic ROI — suppress detections above the grass line ─
        roi_top = self._estimate_field_top(frame)
        x1a, y1a, x2a, y2a = merged.xyxy.T
        cy = (y1a + y2a) / 2                    # center-y of each box
        roi_mask = (
            (cy > H * roi_top) &                # above grass line → reject
            (cy < H * 0.90)                     # below scorebar → reject
        )

        # ── Filter 2: Box size — suppress tiny crowd heads ────────────────────
        widths  = x2a - x1a
        heights = y2a - y1a
        areas   = widths * heights
        size_mask = (
            (areas   > self.cfg.min_box_area_ratio   * H * W) &
            (heights > self.cfg.min_box_height_ratio * H    ) &
            (heights < 0.95 * H)                              # reject impossible full-frame boxes
        )

        final_mask = roi_mask & size_mask
        merged = merged[final_mask]

        return merged

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap, fps, W, H, total = self._open_video(self.cfg.input_path)
        out_fps = fps / self.cfg.frame_skip if self.cfg.frame_skip > 1 else fps
        writer  = self._make_writer(self.cfg.output_path, out_fps, W, H)

        print(f"Video      : {W}x{H} @ {fps:.1f}fps  ({total} frames)")
        print(f"Model      : {self.cfg.model}  conf={self.cfg.conf}  iou={self.cfg.iou}")
        print(f"Frame skip : every {self.cfg.frame_skip} frame(s)")
        print(f"Tiling     : {'ON' if self.cfg.use_tiling else 'OFF'}")
        print(f"Classes    : {self.cfg.classes}  (0=person, 32=sports ball)")
        print(f"Output     : {self.cfg.output_path}")
        print()

        prev_annotated = None
        screenshot_targets = {int(total * f) for f in [0.10, 0.25, 0.50, 0.75, 0.90]}
        start_time = time.time()

        with tqdm(total=total, unit="frame") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                pbar.update(1)

                # Skip frames — write previous annotated frame to keep video length stable
                if frame_idx % self.cfg.frame_skip != 0:
                    if prev_annotated is not None:
                        writer.write(prev_annotated)
                    continue

                # ── Detect ────────────────────────────────────────────────────
                detections = self.detect(frame)

                # ── Track ─────────────────────────────────────────────────────
                tracked = self.tracker.update_with_detections(detections)

                # Safely extract arrays
                if len(tracked) > 0:
                    boxes     = tracked.xyxy
                    track_ids = tracked.tracker_id
                    confs     = tracked.confidence
                    self.all_ids.update(int(t) for t in track_ids)
                else:
                    boxes     = np.empty((0, 4))
                    track_ids = np.empty((0,), dtype=int)
                    confs     = np.empty((0,))

                # ── Log ───────────────────────────────────────────────────────
                if self.cfg.save_json:
                    self.track_log[frame_idx] = [
                        {
                            "id":   int(tid),
                            "bbox": [float(v) for v in box],
                            "conf": float(conf),
                        }
                        for tid, box, conf in zip(track_ids, boxes, confs)
                    ]

                # ── Annotate ──────────────────────────────────────────────────
                annotated = self.annotator.draw_frame(
                    frame,
                    boxes_xyxy=boxes,
                    track_ids=track_ids,
                    confidences=confs,
                    frame_count=frame_idx,
                    total_unique=len(self.all_ids),
                )

                writer.write(annotated)
                prev_annotated = annotated

                # Save screenshots at key moments
                if frame_idx in screenshot_targets:
                    sname = f"outputs/screenshots/frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(sname, annotated)

        cap.release()
        writer.release()

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print(f"\nCompleted in {mins}m {secs}s")
        print(f"Total unique IDs tracked : {len(self.all_ids)}")
        print(f"Annotated video saved    : {self.cfg.output_path}")

        if self.cfg.save_json:
            json_path = "outputs/tracks.json"
            with open(json_path, "w") as f:
                json.dump(self.track_log, f, indent=2)
            print(f"Track log saved          : {json_path}")

        print(f"Screenshots saved        : outputs/screenshots/")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    pipeline = MOTPipeline(cfg)
    pipeline.run()
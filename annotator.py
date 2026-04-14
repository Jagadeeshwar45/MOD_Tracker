"""
Drawing utilities — bounding boxes, IDs, trails, count overlay.
"""
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# Deterministic color per track ID
def id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Returns a stable BGR color for a given track ID."""
    np.random.seed(track_id * 37 + 13)
    color = tuple(int(c) for c in np.random.randint(80, 230, 3))
    return color  # (B, G, R)


class TrackAnnotator:
    """
    Handles all visual annotation on video frames.
    """

    def __init__(self, trail_length: int = 30):
        self.trail_length = trail_length
        self.trail_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    def update_trails(self, track_ids: np.ndarray, boxes_xyxy: np.ndarray):
        """Store center points for trail drawing."""
        current_ids = set(int(t) for t in track_ids)
        # Remove stale trails
        stale = [tid for tid in self.trail_history if tid not in current_ids]
        for tid in stale:
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
        Draw boxes, trails, IDs and a status bar on a frame.

        Args:
            frame:       BGR frame (modified in-place copy)
            boxes_xyxy:  (N, 4) array of [x1, y1, x2, y2]
            track_ids:   (N,) array of integer IDs
            confidences: (N,) detection confidence scores (optional)
            frame_count: current frame index for the HUD
            total_unique: total distinct IDs seen so far

        Returns:
            Annotated BGR frame.
        """
        out = frame.copy()
        self.update_trails(track_ids, boxes_xyxy)

        # Draw trails first (under boxes)
        for tid, pts in self.trail_history.items():
            color = id_to_color(tid)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(2 * alpha))
                cv2.line(out, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # Draw bounding boxes and labels
        for i, (box, tid) in enumerate(zip(boxes_xyxy, track_ids)):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            tid = int(tid)
            color = id_to_color(tid)

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label background
            label = f"ID:{tid}"
            if confidences is not None:
                label += f" {confidences[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx1, ly1 = x1, max(y1 - th - 6, 0)
            lx2, ly2 = x1 + tw + 6, y1
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(
                out, label, (lx1 + 3, ly2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

        # HUD bar
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
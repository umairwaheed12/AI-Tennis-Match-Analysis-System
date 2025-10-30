from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
from .draw import draw_blob_label
class TrackState:
    __slots__ = ("sx","sy","sr","vx","vy","last_frame","locked")
    def __init__(self, x, y, r, frame_idx, locked):
        self.sx, self.sy, self.sr = float(x), float(y), float(r)
        self.vx, self.vy = 0.0, 0.0
        self.last_frame = frame_idx
        self.locked = locked
        

class BallTracker:
    def __init__(self, model_path, conf_enter=0.1, conf_keep=0.08, iou=0.5, imgsz=640,
                 smooth_beta=0.35, max_gap=5, max_step_frac=1/25.0,
                 color=(0,0,255), traj_len=8, circle_thick=2, label=True,
                 ball_class_idx=None, window_title="Tennis Ball - Circle + Trajectory"):
        self.model = YOLO(model_path)
        self.conf_enter = conf_enter
        self.conf_keep  = conf_keep
        self.iou  = iou
        self.imgsz = imgsz
        self.smooth_beta = smooth_beta
        self.max_gap = max_gap
        self.max_step_frac = max_step_frac
        self.color = color
        self.traj_len = traj_len
        self.circle_thick = circle_thick
        self.label = label
        self.ball_class_idx = ball_class_idx

        self.states = {}
        self.trails = defaultdict(lambda: deque(maxlen=self.traj_len))
        self.frame_idx = 0
        self.max_step = None
        self.window_title = window_title

    def _ensure_max_step(self, width):
        if self.max_step is None:
            self.max_step = max(2, int(width * self.max_step_frac))


    def get_last_center_px(self):
        
        if not self.states:
            return None
        best = None
        for st in self.states.values():
            if st.locked and (best is None or st.last_frame > best.last_frame):
                best = st
        if best is None:
            return None
        return (int(best.sx), int(best.sy))

    def stream(self, video_path, width, pre_annotate_fn=None):
        self._ensure_max_step(width)
        for res in self.model.track(
            source=video_path,
            conf=min(self.conf_enter, 0.99),
            imgsz=self.imgsz,
            iou=self.iou,
            tracker="bytetrack.yaml",
            stream=True,
            persist=True,
            verbose=False
        ):
            frame = res.orig_img.copy()
            clean_for_court = pre_annotate_fn(frame.copy()) if callable(pre_annotate_fn) else frame.copy()
            
            
            detections = []

            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                confs      = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.ones(len(boxes_xyxy))
                clss       = res.boxes.cls.cpu().numpy() if res.boxes.cls is not None else np.zeros(len(boxes_xyxy))
                ids        = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.array([-1]*len(boxes_xyxy))

                for (x1, y1, x2, y2), conf, cls, tid in zip(boxes_xyxy, confs, clss, ids):
                    if self.ball_class_idx is not None and int(cls) != self.ball_class_idx:
                        continue
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    r  = 0.5 * min(max(1, x2 - x1), max(1, y2 - y1))
                    detections.append((tid, float(conf), float(cx), float(cy), float(r)))

            det_map = defaultdict(list)
            for tid, conf, cx, cy, r in detections:
                det_map[tid].append((conf, cx, cy, r))
            for tid, lst in det_map.items():
                det_map[tid] = [max(lst, key=lambda t: t[0])]

            for tid, det_list in det_map.items():
                conf, mx, my, mr = det_list[0]
                st = self.states.get(tid)
                if st is None:
                    if conf >= self.conf_enter:
                        st = TrackState(mx, my, mr, self.frame_idx, locked=True)
                        self.states[tid] = st
                else:
                    if st.locked or conf >= self.conf_enter:
                        st.locked = st.locked or (conf >= self.conf_enter)
                    if not st.locked and conf < self.conf_enter:
                        continue

                    px, py, pr = st.sx, st.sy, st.sr
                    sx = (1 - self.smooth_beta) * px + self.smooth_beta * mx
                    sy = (1 - self.smooth_beta) * py + self.smooth_beta * my
                    sr = (1 - self.smooth_beta) * pr + self.smooth_beta * mr
                    dx, dy = sx - px, sy - py
                    mag = (dx*dx + dy*dy) ** 0.5
                    if mag > self.max_step:
                        scale = self.max_step / (mag + 1e-6)
                        sx = px + dx * scale
                        sy = py + dy * scale
                    st.vx, st.vy = sx - px, sy - py
                    st.sx, st.sy, st.sr = sx, sy, sr
                    st.last_frame = self.frame_idx

            for tid, st in list(self.states.items()):
                if tid not in det_map:
                    gap = self.frame_idx - st.last_frame
                    if gap <= self.max_gap and st.locked:
                        sx = st.sx + st.vx
                        sy = st.sy + st.vy
                        st.vx *= 0.9
                        st.vy *= 0.9
                        dx, dy = sx - st.sx, sy - st.sy
                        mag = (dx*dx + dy*dy) ** 0.5
                        if mag > self.max_step:
                            scale = self.max_step / (mag + 1e-6)
                            sx = st.sx + dx * scale
                            sy = st.sy + dy * scale
                        st.sx, st.sy = sx, sy
                    else:
                        if (not st.locked) or gap > self.max_gap:
                            self.states.pop(tid, None)

            # draw
            for tid, st in list(self.states.items()):
                if not st.locked:
                    continue
                cx, cy, r = int(st.sx), int(st.sy), max(2, int(st.sr))
                self.trails[tid].append((cx, cy))
                cv2.circle(frame, (cx, cy), r, self.color, self.circle_thick)
                cv2.circle(frame, (cx, cy), max(2, r // 5), self.color, -1)
                label_y_ref = cy - r - 6  # a little above the ball
                draw_blob_label(frame, "Ball", cx, label_y_ref, fill=(255, 100, 100))  # magenta-ish for strong visibility
                pts = self.trails[tid]
                if len(pts) > 1:
                    for i in range(1, len(pts)):
                        p1, p2 = pts[i - 1], pts[i]
                        thickness = max(1, 3 - i // 4)
                        cv2.line(frame, p1, p2, self.color, thickness)
                if self.label:
                    cv2.putText(frame, f"ID {tid}", (cx + r + 4, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1, cv2.LINE_AA)

            self.frame_idx += 1
            yield frame, clean_for_court
            
    # --- HUD: ball speed in pixels/sec from latest locked state ---
    def current_speed_px_s(self, fps: float) -> float:
        if fps <= 0 or not self.states:
            return 0.0
        # freshest locked
        best = None
        for st in self.states.values():
            if st.locked and (best is None or st.last_frame > best.last_frame):
                best = st
        if best is None:
            return 0.0
        return ((best.vx**2 + best.vy**2) ** 0.5) * fps

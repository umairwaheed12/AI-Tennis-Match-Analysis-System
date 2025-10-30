from ultralytics import YOLO
import numpy as np
from collections import deque
from .draw import draw_box_with_label
from .draw import draw_box_with_label, draw_blob_label
import cv2
# Colors (same as before)
PLAYER_COLOR   = (0, 200, 0)
OFFICIAL_COLOR = (0, 0, 255)

def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    return inter / max(1e-6, (areaA + areaB - inter))

def _center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)*0.5, (y1+y2)*0.5)

def _center_dist(a, b):
    ax, ay = _center(a)
    bx, by = _center(b)
    dx, dy = ax-bx, ay-by
    return (dx*dx + dy*dy) ** 0.5

class _Track:
    __slots__ = ("slot_id","cls","box","score","miss","hist","prev_c","speed_px_s")
    def __init__(self, slot_id, cls, box, score):
        self.slot_id = slot_id  # fixed slot number (1 or 2 for players; 1 for official)
        self.cls = cls
        self.box = box          # (x1,y1,x2,y2)
        self.score = float(score)
        self.miss = 0
        self.hist = deque(maxlen=5)
        self.prev_c = None          # last center (x,y)
        self.speed_px_s = 0.0       # current speed in pixels/second

class PeopleOverlay:
    
    def __init__(self, model_path, conf=0.30, imgsz=960,
             player_idx=1, official_idx=0,
             iou_match=0.35, max_miss=12, ema=0.6, center_gate_px=120,
             pose_model_path=None, pose_conf=0.35, pose_imgsz=640, show_player_skeleton=True):
        # detector (existing)
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.player_idx = int(player_idx)
        self.official_idx = int(official_idx)

        # tracking params (existing)
        self.iou_match = float(iou_match)
        self.max_miss  = int(max_miss)
        self.ema = float(ema)
        self.center_gate_px = float(center_gate_px)

        # fixed slots + HUD state (existing)
        self.player_slots = [None, None]
        self.official_slot = None
        self.last_player_speed = {1: 0.0, 2: 0.0}
        self.official_present = False

        # --- NEW: optional pose model for skeletons ---
        self.pose_model = YOLO(pose_model_path) if pose_model_path else None
        self.pose_conf = float(pose_conf)
        self.pose_imgsz = int(pose_imgsz)
        self.show_player_skeleton = bool(show_player_skeleton)
        # after you've decided kps_to_draw for player slot 'si'
        # placeholders for 3D view
        self.last_player_kps = [None, None]
        self.last_player_box = [None, None]
        self.last_ball_xy = None
                

    def _smooth(self, prev_box, new_box):
        if prev_box is None: return new_box
        a = self.ema
        return (
            a*prev_box[0] + (1-a)*new_box[0],
            a*prev_box[1] + (1-a)*new_box[1],
            a*prev_box[2] + (1-a)*new_box[2],
            a*prev_box[3] + (1-a)*new_box[3],
        )

    def _match_single(self, track, dets):
        
        best_i, best_s = -1, -1.0
        for i, (box, score) in enumerate(dets):
            iou = _iou(track.box, box)
            if iou < self.iou_match:
                continue
            if self.center_gate_px > 0 and _center_dist(track.box, box) > self.center_gate_px:
                continue
            # score by (IoU + small confidence weight)
            s = iou + 0.05 * score
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    def _update_slot(self, slot, cls_idx, dets, fps):
        if slot is None:
            return -1
        j = self._match_single(slot, dets)
        if j >= 0:
            box, score = dets[j]
            # ---- speed (px/s) from center movement ----
            new_c = _center(box)
            if slot.prev_c is not None and fps > 0:
                dx = new_c[0] - slot.prev_c[0]
                dy = new_c[1] - slot.prev_c[1]
                slot.speed_px_s = ((dx*dx + dy*dy) ** 0.5) * fps
            slot.prev_c = new_c
            # -------------------------------------------

            slot.box = self._smooth(slot.box, box)
            slot.score = max(slot.score, float(score))
            slot.miss = 0
            return j
        else:
            slot.miss += 1
            slot.speed_px_s *= 0.9   # slow decay when missed
            return -1

    
    def _bbox_of_keypoints(self, kps_xy):
        
        if kps_xy is None or len(kps_xy) == 0:
            return None
        xs = kps_xy[:, 0][~np.isnan(kps_xy[:, 0])]
        ys = kps_xy[:, 1][~np.isnan(kps_xy[:, 1])]
        if xs.size == 0 or ys.size == 0:
            return None
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    def _match_pose_to_box(self, det_box, poses, disallow=None):
        
        best_i, best_s = -1, -1.0
        if not poses:
            return -1
        for i, P in enumerate(poses):
            if disallow and i in disallow:
                continue
            pb = P.get("bbox")
            if pb is None:
                continue
            iou = _iou(det_box, pb)
            if iou > best_s:
                best_s, best_i = iou, i
        return best_i

    def _draw_player_skeleton(self, frame, kps_xy, color=(0, 255, 0), th=2):
        
        if kps_xy is None:
            return
        pts = []
        for x, y in kps_xy:
            if np.isnan(x) or np.isnan(y):
                pts.append(None)
            else:
                pts.append((int(x), int(y)))

        # Typical pairs for 17-kp COCO order (edit if your model differs)
        pairs = [
            (5,7), (7,9),       # left arm
            (6,8), (8,10),      # right arm
            (11,13), (13,15),   # left leg
            (12,14), (14,16),   # right leg
            (5,6),              # shoulders
            (11,12),            # hips
            (5,11), (6,12)      # torso diagonals (stability)
        ]
        for a, b in pairs:
            if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                cv2.line(frame, pts[a], pts[b], color, th, lineType=cv2.LINE_AA)
        for p in pts:
            if p:
                cv2.circle(frame, p, 3, color, -1, lineType=cv2.LINE_AA)

    def _spawn_into_empty_player_slot(self, dets_sorted):
        
        # which slots are usable?
        usable = []
        for idx in range(2):
            t = self.player_slots[idx]
            if t is None or t.miss > self.max_miss:
                usable.append(idx)
        if not usable or not dets_sorted:
            return
        slot_idx = usable[0]
        box, score = dets_sorted[0]
        # fixed IDs: slot 0 -> Player #1, slot 1 -> Player #2
        self.player_slots[slot_idx] = _Track(slot_id=slot_idx+1, cls=self.player_idx, box=box, score=score)

    def _spawn_official_if_empty(self, best_off):
        if self.official_slot is None or self.official_slot.miss > self.max_miss:
            if best_off is not None:
                box, score = best_off
                # Official fixed ID: #1
                self.official_slot = _Track(slot_id=1, cls=self.official_idx, box=box, score=score)
                
    def _pose_on_box(self, src, box):
        
        if self.pose_model is None:
            return None
        x1, y1, x2, y2 = map(int, box)
        h, w = src.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = src[y1:y2, x1:x2]
        pres = self.pose_model.predict(
            source=crop, imgsz=self.pose_imgsz, conf=max(0.05, self.pose_conf * 0.8),  # a bit looser on crop
            verbose=False
        )
        if pres and pres[0] is not None and getattr(pres[0], "keypoints", None) is not None and len(pres[0].keypoints.xy) > 0:
            k = pres[0].keypoints.xy[0].cpu().numpy()  # (K,2)
            # translate back to full-frame
            k[:, 0] += x1
            k[:, 1] += y1
            return k
        return None


    def annotate(self, frame, infer_src=None, fps=30):
        
        src = infer_src if infer_src is not None else frame

        # --- 1) Detect ---
        res = self.model.predict(source=src, imgsz=self.imgsz, conf=self.conf, verbose=False)
        p_dets, o_dets = [], []   # lists of (box, score)
        if res and res[0].boxes is not None:
            b = res[0].boxes
            xyxy = b.xyxy.cpu().numpy()
            clss = b.cls.cpu().numpy() if b.cls is not None else np.zeros(len(xyxy))
            confs = b.conf.cpu().numpy() if b.conf is not None else np.ones(len(xyxy))
            for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
                cls = int(cls)
                box = (float(x1), float(y1), float(x2), float(y2))
                if cls == self.player_idx:
                    p_dets.append((box, float(c)))
                elif cls == self.official_idx:
                    o_dets.append((box, float(c)))

        # sort remaining by confidence (desc) for spawn decisions
        p_dets.sort(key=lambda t: t[1], reverse=True)
        o_dets.sort(key=lambda t: t[1], reverse=True)
        best_off = o_dets[0] if o_dets else None

        # --- 2) Update existing slots (players) ---
        used_p = set()
        for si in range(2):
            t = self.player_slots[si]
            if t is None:
                continue
            # pass fps through to the tracker update
            idx = self._update_slot(t, self.player_idx, p_dets, fps)
            if idx >= 0:
                used_p.add(idx)

        # prune long-missed player slots
        for si in range(2):
            t = self.player_slots[si]
            if t is not None and t.miss > self.max_miss:
                self.player_slots[si] = None

        # --- 3) Spawn into empty player slots (use remaining best) ---
        p_remaining = [p_dets[i] for i in range(len(p_dets)) if i not in used_p]
        self._spawn_into_empty_player_slot(p_remaining)

        # --- 4) Update / spawn official slot ---
        if self.official_slot is not None:
            _ = self._update_slot(self.official_slot, self.official_idx, o_dets, fps)
            if self.official_slot is not None and self.official_slot.miss > self.max_miss:
                self.official_slot = None
        self._spawn_official_if_empty(best_off)

        # (optional) expose HUD fields if you added them in __init__
        if hasattr(self, "last_player_speed"):
            self.last_player_speed = {1: 0.0, 2: 0.0}
            for si in range(2):
                t = self.player_slots[si]
                if t is not None and hasattr(t, "speed_px_s"):
                    self.last_player_speed[t.slot_id] = float(t.speed_px_s)
        if hasattr(self, "official_present"):
            self.official_present = self.official_slot is not None
            
        poses = []
        # 🔹 Only run pose detection on player crops, not the whole frame
        if self.pose_model is not None and self.show_player_skeleton:
            for si in range(2):  # two player slots only
                t = self.player_slots[si]
                if t is None:
                    continue
                kps = self._pose_on_box(src, t.box)
                if kps is not None:
                    bbox = self._bbox_of_keypoints(kps)
                    poses.append({"bbox": bbox, "kps": kps})

        
        # --- 5) Draw (persistent, fixed IDs) ---
        # Player slots: fixed IDs #1 and #2
        if not hasattr(self, "_used_pose_idx"):
            self._used_pose_idx = set()

        for si in range(2):
            t = self.player_slots[si]
            if t is None:
                continue
            x1, y1, x2, y2 = t.box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), PLAYER_COLOR, 3)
            cx = int((x1 + x2) * 0.5)
            draw_blob_label(frame, f"Player {t.slot_id}", cx, int(y1), fill=PLAYER_COLOR)

            # --- match pose and draw skeleton (players only) ---
            if self.pose_model is not None and self.show_player_skeleton:
                kps_to_draw = None

                # Fast path: match from global 'poses'
                if poses:
                    pi = self._match_pose_to_box(t.box, poses, disallow=self._used_pose_idx)
                    if pi >= 0 and poses[pi].get("kps") is not None:
                        self._used_pose_idx.add(pi)
                        kps_to_draw = poses[pi]["kps"]

                # Fallback: run pose on this player's crop if we still don't have kps
                if kps_to_draw is None:
                    kps_to_draw = self._pose_on_box(src, t.box)

                if kps_to_draw is not None:
                    self._draw_player_skeleton(frame, kps_to_draw, color=(0, 255, 255), th=2)
                    # Save for 3D visualization
                    self.last_player_kps[si] = kps_to_draw
                    self.last_player_box[si] = t.box

        # ✅ now clear once per frame (outside the loop)
        self._used_pose_idx.clear()

                


        # Official: fixed ID #1
        if self.official_slot is not None:
            x1, y1, x2, y2 = self.official_slot.box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), OFFICIAL_COLOR, 3)
            # pill stays:
            cx = int((x1 + x2) * 0.5)
            draw_blob_label(frame, "Official", cx, int(y1), fill=OFFICIAL_COLOR)
            
            
            
            

        return frame

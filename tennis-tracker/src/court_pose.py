from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# same connections you used
LINE_CONNECTIONS = [
    (1, 3),
    (2, 'br'),
    (1, 2),
    (3, 'br'),
    (5, 6),
    (7, 10), (10, 12), (12, 8),
    (13, 14), (13, 10),
    (11, 14), (14, 12)
]

def _ema(prev, new, alpha):
    if prev is None or new is None:
        return new
    x = int(alpha * prev[0] + (1 - alpha) * new[0])
    y = int(alpha * prev[1] + (1 - alpha) * new[1])
    return (x, y)

def _median_point(points):
    
    xs, ys = [], []
    for p in points:
        if p is not None:
            xs.append(p[0]); ys.append(p[1])
    if not xs:
        return None
    return (int(np.median(xs)), int(np.median(ys)))

def _dist(a, b):
    if a is None or b is None: return 0.0
    dx, dy = a[0]-b[0], a[1]-b[1]
    return (dx*dx + dy*dy) ** 0.5

class CourtPose:
    
    def __init__(self, model_path, conf=0.5, imgsz=640,
                 ema_alpha=0.85,                 # a bit stronger than 0.7 to reduce lag+jitter
                 median_win=5,                   # short history for temporal median
                 outlier_px=6,                   # clamp sudden jumps
                 min_visible_frac=0.5):          # if <50% kps valid, hold last
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

        self.ema_alpha = float(ema_alpha)
        self.outlier_px = int(outlier_px)
        self.min_visible_frac = float(min_visible_frac)

        # History: list of keypoint lists (length K), each as deque of recent points
        self.kp_hist = None       # will init when first kp size known
        self.kp_hist_len = int(median_win)

        self.prev_keypoints = None
        self.prev_br = None
        self.br_hist = deque(maxlen=self.kp_hist_len)
        self.scale_px_per_m = None     # pixels per meter
        self.scale_alpha = 0.85        # smoothing for scale
        # Indices (1-based) of near-baseline singles corners (adjust if your model differs)
        self.scale_left_idx = 8        # bottom-left singles corner (yellow "8" in your overlay)
        self.scale_right_idx = 16      # bottom-right singles corner (yellow "16")
        self.SINGLES_WIDTH_M = 8.23    # official singles width (meters)

    def _ensure_hist(self, k):
        if self.kp_hist is None:
            self.kp_hist = [deque(maxlen=self.kp_hist_len) for _ in range(k)]

    def _stabilize_keypoints(self, raw_kps):
        
        if raw_kps is None:
            return self.prev_keypoints

        K = len(raw_kps)
        self._ensure_hist(K)

        # push into per-kp history
        for i in range(K):
            self.kp_hist[i].append(raw_kps[i])

        # median over window
        med = [ _median_point(list(self.kp_hist[i])) for i in range(K) ]

        # EMA on top of median
        smoothed = []
        for i in range(K):
            sm = _ema(self.prev_keypoints[i] if self.prev_keypoints else None,
                      med[i], self.ema_alpha)
            # outlier clamp vs previous smoothed
            if self.prev_keypoints and sm is not None:
                if _dist(sm, self.prev_keypoints[i]) > self.outlier_px:
                    # pull back toward previous to avoid jitter spikes
                    px, py = self.prev_keypoints[i]
                    mx = px + (sm[0] - px) * min(1.0, self.outlier_px / max(1.0, _dist(sm, self.prev_keypoints[i])))
                    my = py + (sm[1] - py) * min(1.0, self.outlier_px / max(1.0, _dist(sm, self.prev_keypoints[i])))
                    sm = (int(mx), int(my))
            smoothed.append(sm)
        return smoothed

    def _stabilize_br(self, br):
        self.br_hist.append(br)
        med = _median_point(list(self.br_hist))
        sm  = _ema(self.prev_br, med, self.ema_alpha)
        if self.prev_br and sm is not None and _dist(sm, self.prev_br) > self.outlier_px:
            px, py = self.prev_br
            mx = px + (sm[0] - px) * min(1.0, self.outlier_px / max(1.0, _dist(sm, self.prev_br)))
            my = py + (sm[1] - py) * min(1.0, self.outlier_px / max(1.0, _dist(sm, self.prev_br)))
            sm = (int(mx), int(my))
        return sm

    def _update_scale_from_keypoints(self, keypoints):
        
        if not keypoints:
            return

        # Candidate pairs: near baseline (bottom), far baseline (top).
        # Adjust indices if your model uses different IDs for these corners.
        pairs = [
            (8, 16),   # bottom-left singles corner, bottom-right singles corner  (your overlay shows 8 & 16)
            (5, 7),    # top-left singles corner,    top-right singles corner
        ]

        px_per_m_now = None
        for li1, ri1 in pairs:
            li, ri = li1 - 1, ri1 - 1
            if li < 0 or ri < 0 or li >= len(keypoints) or ri >= len(keypoints):
                continue
            pL, pR = keypoints[li], keypoints[ri]
            if not pL or not pR:
                continue
            dx = pL[0] - pR[0]
            dy = pL[1] - pR[1]
            dist_px = (dx*dx + dy*dy) ** 0.5
            if dist_px > 1.0:
                px_per_m_now = dist_px / self.SINGLES_WIDTH_M
                break

        if px_per_m_now is None:
            # nothing visible this frame; keep last stable value
            return

        if self.scale_px_per_m is None:
            self.scale_px_per_m = px_per_m_now
        else:
            a = self.scale_alpha
            self.scale_px_per_m = a * self.scale_px_per_m + (1 - a) * px_per_m_now


    def px_per_s_to_kmh(self, v_px_s: float) -> float:
        
        if self.scale_px_per_m is None or v_px_s <= 0:
            return 0.0
        m_per_s = v_px_s / self.scale_px_per_m
        return m_per_s * 3.6
    
    def annotate(self, frame, infer_src=None):
        results = self.model.predict(
            source=infer_src if infer_src is not None else frame,
            conf=self.conf, imgsz=self.imgsz, verbose=False
        )
        annotated = frame.copy()

        for result in results:
            # --- raw keypoints ---
            raw_kps = []
            if result.keypoints is not None:
                kps = result.keypoints.xy[0].cpu().numpy()
                for (x, y) in kps:
                    if np.isnan(x) or np.isnan(y):
                        raw_kps.append(None)
                    else:
                        raw_kps.append((int(x), int(y)))

            # visibility check
            valid = sum(1 for p in raw_kps if p is not None)
            total = max(1, len(raw_kps))
            visible_frac = valid / total

            # --- stabilize or hold ---
            if visible_frac < self.min_visible_frac and self.prev_keypoints is not None:
                keypoints = self.prev_keypoints[:]  # hold last stable
            else:
                keypoints = self._stabilize_keypoints(raw_kps)

            self.prev_keypoints = keypoints
            self._update_scale_from_keypoints(keypoints)

            # --- bottom-right (BR) corner from box ---
            br_point = None
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[0].cpu().numpy())
                br_point = (x2, y2)

            if br_point is None and self.prev_br is not None:
                br_point = self.prev_br  # hold last
            else:
                br_point = self._stabilize_br(br_point)

            self.prev_br = br_point

            

            # --- draw red lines first ---
            if keypoints:
                for conn in LINE_CONNECTIONS:
                    p1, p2 = conn
                    if p1 == 'br' or p2 == 'br':
                        if br_point is None:
                            continue
                        if p1 == 'br' and isinstance(p2, int) and keypoints[p2 - 1]:
                            cv2.line(annotated, br_point, keypoints[p2 - 1], (0, 0, 255), 3)
                        elif p2 == 'br' and isinstance(p1, int) and keypoints[p1 - 1]:
                            cv2.line(annotated, keypoints[p1 - 1], br_point, (0, 0, 255), 3)
                    else:
                        if (p1 <= len(keypoints) and p2 <= len(keypoints)
                                and keypoints[p1 - 1] and keypoints[p2 - 1]):
                            cv2.line(annotated, keypoints[p1 - 1], keypoints[p2 - 1], (0, 0, 255), 3)

            # --- draw mirror point (yellow instead of green) ---
            if keypoints and len(keypoints) >= 13 and keypoints[9] and keypoints[12]:
                x13, y13 = keypoints[12]
                x10, y10 = keypoints[9]
                mirror_x = int(2 * x13 - x10)
                mirror_y = int(2 * y13 - y10)
                mirror_point = (mirror_x, mirror_y)
                cv2.circle(annotated, mirror_point, 7, (0, 255, 255), -1)  # yellow
                cv2.putText(annotated, "15", (mirror_x + 8, mirror_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(annotated, keypoints[12], mirror_point, (0, 0, 255), 3)
                cv2.line(annotated, keypoints[9], keypoints[12], (0, 0, 255), 3)
                
            # --- draw BR point (yellow instead of green) ---
            if br_point is not None:
                cv2.circle(annotated, br_point, 7, (0, 255, 255), -1)  # yellow
                cv2.putText(annotated, "16", (br_point[0] + 10, br_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            for i, pt in enumerate(keypoints or [], start=1):
                if pt:
                    # Draw the yellow dot above red lines
                    cv2.circle(annotated, pt, 6, (0, 255, 255), -1)
                    
                    # Commented out numbering (keep for later if needed)
                    cv2.putText(annotated, str(i), (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    
            # --- cache per-ID court points for homography ---
            # use the stabilized keypoints we just computed in this frame
            self._last_court_pts_by_id = {}

            if keypoints is not None and len(keypoints) > 0:
                # keypoints is 1-based in your overlay (we draw 1..N),
                # so store them with those same IDs.
                for i, p in enumerate(keypoints, start=1):
                    if p is None:
                        continue
                    x, y = float(p[0]), float(p[1])
                    self._last_court_pts_by_id[i] = (x, y)

            # ensure BR is present as id 16 (you call it "br")
            if br_point is not None:
                self._last_court_pts_by_id[16] = (float(br_point[0]), float(br_point[1]))
                
                
        return annotated
    
    # src/court_pose.py  → get_four_corners_px()
    # --- REPLACE the whole get_four_corners_px() in src/court_pose.py ---

    # --- REPLACE the whole get_four_corners_px() in src/court_pose.py ---

    def get_four_corners_px(self):
        
        import numpy as np
        d = getattr(self, "_last_court_pts_by_id", None)
        if not d:
            return None

        need = [1, 2, 16, 3]  # TL, TR, BR, BL
        pts = []
        for pid in need:
            p = d.get(pid)
            if p is None:
                return None
            pts.append(p)
        return np.asarray(pts, dtype=np.float32)



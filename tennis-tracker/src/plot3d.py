# src/plot3d.py
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa
from .lifter3d import PoseLifter3D
from matplotlib.gridspec import GridSpec
# Standard singles tennis court in meters (centered around origin for symmetry)
# We'll draw on the X-Y plane (Z up).
COURT_LEN = 23.77      # baseline to baseline
COURT_WID = 8.23       # singles width
SERV_BOX_LEN = 6.40    # service line distance from net
NET_X = 0.0            # net at X=0 if we center court

# skeleton graph (COCO-ish 17 keypoints) – pairs of indices
SKELETON_PAIRS = [
    (5,7), (7,9),
    (6,8), (8,10),
    (11,13), (13,15),
    (12,14), (14,16),
    (5,6),
    (11,12),
    (5,11), (6,12)
]

class Tennis3DPlot:
    def __init__(self, fig_size=(5, 5), dpi=120, bg_color=(0, 0, 0)):
        
        
        self.fig = Figure(figsize=(10.0, 11.0), dpi=dpi,
                  facecolor=(bg_color[0]/255, bg_color[1]/255, bg_color[2]/255))
        self.canvas = FigureCanvasAgg(self.fig)

        # top: 2D court  |  bottom: 3D scene
        gs = GridSpec(2, 1, height_ratios=[0.47, 0.53], hspace=0.04, figure=self.fig)

        # --- 2D COURT AXIS ---
        self.ax2d = self.fig.add_subplot(gs[0])
        self.ax2d.set_facecolor('k')
        halfL = COURT_LEN / 2
        halfW = COURT_WID / 2
        self.ax2d.set_xlim(-halfL, halfL)
        self.ax2d.set_ylim(-halfW, halfW)
        self.ax2d.set_aspect('equal', adjustable='box')
        self.ax2d.set_xticks([]); self.ax2d.set_yticks([])
        for sp in self.ax2d.spines.values():
            sp.set_color((0.4, 0.4, 0.4))

        # 3D AXIS BELOW
        self.ax = self.fig.add_subplot(gs[1], projection='3d', facecolor='k')
        self.ax.view_init(elev=25, azim=-60)

        # slightly shrink entire 3D view (reduces vertical height)
        box = self.ax.get_position()
        self.ax.set_position([box.x0 - 0.10,  # small nudge left
                      box.y0 - 0.15,  # **more downward shift**
                      box.width * 1.40,
                      box.height * 1.35])  # slight shrink vs before to avoid overlap
        

        # camera angle
        self.ax.view_init(elev=25, azim=-60)

        # axes limits (meters)
        halfL = COURT_LEN / 2.0
        halfW = COURT_WID / 2.0
        self.ax.set_xlim(-halfL, halfL)
        self.ax.set_ylim(-halfW, halfW)
        self.ax.set_zlim(0, 3.0)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        # pane/grid styling (old/new mpl APIs)
        try:
            self.ax.w_xaxis.set_pane_color((0, 0, 0, 1))
            self.ax.w_yaxis.set_pane_color((0, 0, 0, 1))
            self.ax.w_zaxis.set_pane_color((0, 0, 0, 1))
        except AttributeError:
            self.ax.xaxis.pane.set_facecolor((0, 0, 0, 1))
            self.ax.yaxis.pane.set_facecolor((0, 0, 0, 1))
            self.ax.zaxis.pane.set_facecolor((0, 0, 0, 1))
        self.ax.grid(True, color=(0.35, 0.35, 0.35))
        

        self.scale_x = 0.50
        self.scale_y = 0.75
        self.offset_xy = (-0.45, -0.25)
        self.court_color = (1.0, 0.0, 0.0)
        self.width_scale_2d = 5.5
        
       # --- 3D skeleton appearance/scale ---
        self.skel_scale    = 0.18   # was 0.30 → smaller XY footprint
        self.skel_max_r_m  = 0.45   # was 0.55 → tighter clamp
        self.skel_lw       = 1.2    # was 1.6  → thinner lines
        self.skel_pt       = 7      # was 10   → smaller joints
        self.skel_z_scale  = 0.55   # NEW: compress height so it doesn't look like a pole


        # dynamic holders
        self.player_lines  = [[], []]
        self.player_joints = [[], []]
        self.ball_scatter  = None
        # --- ball vertical physics (replaces speed->height heuristic) ---
        self.ball_z   = 0.0     # meters
        self.ball_vz  = 0.0     # m/s
        self.g        = 9.81    # gravity
        self.bounce_e = 0.58    # coefficient of restitution (0.5–0.7 looks good)
        self.hit_boost_k = 0.35 # how much planar speed turns into upward vz on hits
        self.hit_boost_c = 1.2  # base upward kick (m/s)
        self.min_bounce_vz = 0.9  # stop bouncing when vz is smaller than this on impact

        # for optional 2D bounce ring
        self._bounce_flash_t = 0.0   # seconds remaining for bounce flash
        self._bounce_ring = None
        self._prev_ball_xy_m = None   # already have this
        # calibration + constants
        # calibration + constants
        self.H        = None
        self.court_W  = COURT_WID   # <<< was 10.973; use singles width 8.23 to match drawn court
        self.court_L  = COURT_LEN   # keep
        self.allow_outside = True  # let players/ball go slightly beyond court outline
        # timing for ball altitude vs speed
        self.fps = 30.0
        self._prev_ball_xy_m = None
        self._ball_alt  = 0.8
        self._ball_alpha = 0.35
        # If the 4 input corners are the INNER service-box, turn this on.
        self.uses_inner_box = False            # set False later if you switch to outer-court corners
        self._x_stretch = COURT_LEN / (2.0 * SERV_BOX_LEN)   # 23.77 / (2*6.40) ≈ 1.8586
        self._y_stretch = COURT_WID / (COURT_WID / 2.0)      # 8.23 / 4.115 = 2.0
        self._use_service_box_H = True  # set True if you passed inner-box corners; False if you pass outer corners
        # --- tunables (in __init__ after court_color) ---
        # --- 3D pose appearance (add in __init__ right after court_color) ---
        self.pose_xy_gain   = 0.85   # how much of the 2D limb spread to keep in 3D (XY)
        self.pose_xy_clip_m = 1.10   # max radius from the feet anchor (meters)
        self.skel_z_scale   = 0.50   # compress height so it’s not a pole
        self.z_profile_gain = 0.30   # small forward/back depth for arms/legs
        # in Tennis3DPlot.__init__
        self.lifter = PoseLifter3D(backend="dummy", person_height_m=1.78)  # swap to your real model later
        self.pose_ema = [None, None]  # smooth per player
        # now it is safe to draw the static court
        self._ax2d_dyn = []
        self._draw_2d_court_lines()
        self._draw_court()
        
    def _snap_if_near_outer_edge_m(self, x, y, tol=0.12):
        
        halfL = self.court_L / 2.0
        halfW = self.court_W / 2.0
        xmin, xmax = -halfL, halfL
        ymin, ymax = -halfW, halfW

        # distances to the four edges
        d_left   = abs(x - xmin)
        d_right  = abs(x - xmax)
        d_bottom = abs(y - ymin)
        d_top    = abs(y - ymax)

        dmin = min(d_left, d_right, d_bottom, d_top)
        if dmin > tol:
            return x, y  # not close → leave unchanged

        # snap to the closest edge
        if dmin == d_left:   x = xmin
        elif dmin == d_right: x = xmax
        elif dmin == d_bottom: y = ymin
        else:                 y = ymax
        return x, y    
    def _clamp_inside_court_m(self, x, y, margin=0.05):
        
        L, W = self.court_L, self.court_W
        xmin, xmax = -L/2 + margin,  L/2 - margin
        ymin, ymax = -W/2 + margin,  W/2 - margin
        return max(xmin, min(xmax, x)), max(ymin, min(ymax, y))    
    def _draw_2d_court_lines(self):
        L, W = self.court_L, self.court_W
        s = getattr(self, "width_scale_2d", 1.0)

        self.ax2d.set_facecolor((0.0, 0.1, 0.5))
        self.ax2d.set_aspect('equal', adjustable='box')
        self.ax2d.set_xticks([]); self.ax2d.set_yticks([])
        self.ax2d.set_anchor('C')

        pad = 7.30
        # x-axis shows WIDTH (rotated), so scale it; y-axis (LENGTH) stays unchanged
        self.ax2d.set_xlim(- (W/2)*s - pad, (W/2)*s + pad)
        self.ax2d.set_ylim(- L/2      - pad,  L/2      + pad)

        for (a, b) in self._court_segments_m():
            (x1, y1), (x2, y2) = a, b
            # rotate AND widen width dimension only
            self.ax2d.plot([s*y1, s*y2], [x1, x2], color='white', lw=2.2, solid_capstyle='round')



    def _draw_rect(self, x1, x2, y1, y2, color='r', lw=2):
        xs = [x1, x2, x2, x1, x1]
        ys = [y1, y1, y2, y2, y1]
        zs = [0, 0, 0, 0, 0]
        self.ax.plot(xs, ys, zs, color=color, lw=lw)
    def _ema3d(self, prev, cur, a=0.75):
        if prev is None or cur is None: return cur
        return a * prev + (1.0 - a) * cur
    def _court_segments_m(self):
        
        L = COURT_LEN
        W = COURT_WID
        S = SERV_BOX_LEN
        lines = []

        # outer rectangle (singles)
        lines += [
            ((-L/2, -W/2), (-L/2,  W/2)),  # left sideline
            (( L/2, -W/2), ( L/2,  W/2)),  # right sideline
            ((-L/2, -W/2), ( L/2, -W/2)),  # near baseline
            ((-L/2,  W/2), ( L/2,  W/2)),  # far baseline
        ]
        # net line (center)
        lines += [((0, -W/2), (0, W/2))]

        # service boxes
        lines += [
            ((-S, -W/4), (-S,  W/4)),
            (( S, -W/4), ( S,  W/4)),
            ((-S, -W/4), ( S, -W/4)),
            ((-S,  W/4), ( S,  W/4)),
        ]
        return lines
    def _baseline_snap_for_2d(self, x_m, y_m, margin=0.15):
        
        L, W = self.court_L, self.court_W
        # clamp y
        y_m = max(-W/2 + margin, min(W/2 - margin, y_m))
        # choose side by current x: x<0 → left baseline, x>=0 → right baseline
        if x_m < 0:
            x_m = -L/2 + margin   # left baseline center
        else:
            x_m =  L/2 - margin   # right baseline center
        return x_m, y_m
    
    def _draw_court(self):
        z_plane = 0.0
        for (a, b) in self._court_segments_m():
            self._court_line(a, b, z_plane)

    def _load_2d_court_image(self, path):
        
        import numpy as np
        import matplotlib.image as mpimg

        L, W = COURT_LEN, COURT_WID
        tgt_ar = L / W  # 2.886...

        img = mpimg.imread(path).astype(np.float32)  # 0..1
        # → RGBA
        if img.ndim == 2:
            img = np.dstack([img, img, img, np.ones_like(img)])
        elif img.shape[-1] == 3:
            img = np.dstack([img, np.ones(img.shape[:2], dtype=np.float32)])

        rgb, alpha = img[..., :3], img[..., 3]

        # recolor strokes to white (keep transparency)
        mask = (alpha > 0.05) if alpha.max() > 0 else (rgb.mean(axis=-1) < 0.5)
        rgb[mask] = 1.0
        if alpha.max() == 0.0:
            alpha = np.where(mask, 1.0, 0.0)
        img = np.dstack([rgb, alpha])

        # --- PAD to target aspect (letterbox with transparent pixels) ---
        h, w = img.shape[:2]
        ar = w / h
        if abs(ar - tgt_ar) > 1e-3:
            if ar < tgt_ar:
                # too tall → pad left/right
                new_w = int(round(h * tgt_ar))
                pad = new_w - w
                lpad = pad // 2
                rpad = pad - lpad
                pad_lr = ((0, 0), (lpad, rpad), (0, 0))
                img = np.pad(img, pad_lr, mode='constant', constant_values=0.0)
            else:
                # too wide → pad top/bottom
                new_h = int(round(w / tgt_ar))
                pad = new_h - h
                tpad = pad // 2
                bpad = pad - tpad
                pad_tb = ((tpad, bpad), (0, 0), (0, 0))
                img = np.pad(img, pad_tb, mode='constant', constant_values=0.0)

        # draw PNG a bit larger than the court coordinates
        scale = 2.15
        x_half = (L/2) * scale
        y_half = (W/2) * scale

        self.ax2d.imshow(
            img,
            extent=[-x_half, x_half, -y_half, y_half],
            origin='lower',
            aspect='equal',
            zorder=0
        )

        # Blue padding (background)
        self.ax2d.set_facecolor((0.05, 0.15, 0.6))

        # === Center the PNG within the padding and keep a uniform border ===
        border = 0.12  # 12% blue border around the court, tweak 0.08–0.20 to taste
        xv = (L/2) * (1 - border)
        yv = (W/2) * (1 - border)
        self.ax2d.set_xlim(-xv, xv)
        self.ax2d.set_ylim(-yv, yv)
        # ⬆️ nudge the visible window upward by a small fraction of court height
        y_bias = -0.20 * (W/2)     # try 0.08–0.15
        self.ax2d.set_ylim(-yv + y_bias, yv + y_bias)

        # Keep proportions and center the axes content in the subplot
        self.ax2d.set_aspect('equal', adjustable='box')
        self.ax2d.set_anchor('C')   # <-- ensures the content is centered
        self.ax2d.set_xticks([]); self.ax2d.set_yticks([])




    def set_fps(self, fps: float):
        if fps and fps > 1:
            self.fps = float(fps)
            
    
    def _to_plot_xy(self, mx, my):
       
        return (self.scale_x * mx + self.offset_xy[0],
                self.scale_y * my + self.offset_xy[1])

    def _court_line(self, a, b, z):
        
        (x1, y1), (x2, y2) = a, b
        X1, Y1 = self._to_plot_xy(x1, y1)
        X2, Y2 = self._to_plot_xy(x2, y2)
        return self.ax.plot([X1, X2], [Y1, Y2], [z, z],
                            color=self.court_color, linewidth=2.2)
    def _clear_dynamic(self):
        # Remove old skeletons
        for i in range(2):
            for ln in self.player_lines[i]:
                ln.remove()
            self.player_lines[i] = []
            for pt in self.player_joints[i]:
                pt.remove()
            self.player_joints[i] = []
        # Remove ball
        if self.ball_scatter is not None:
            self.ball_scatter.remove()
            self.ball_scatter = None
            
        # track 2D dynamic scatters
        if hasattr(self, "_ax2d_dyn"):
            for h in self._ax2d_dyn: h.remove()
        self._ax2d_dyn = []

    def set_homography_from_four_corners(self, pts_img_px):
        
        if pts_img_px is None or len(pts_img_px) != 4:
            return
        # basic sanity: points should be distinct
        P = np.asarray(pts_img_px, dtype=np.float32).reshape(4, 2)
        if np.min(np.linalg.norm(P - P[:,None,:], axis=2) + np.eye(4)*1e9) < 5.0:
            # some corners are nearly identical -> reject
            return

        L, W = self.court_L, self.court_W
        # Full singles rectangle (outer court), not inner box
        dst = np.float32([
            [-L/2,  W/2],   # TL (far-left)
            [ L/2,  W/2],   # TR (far-right)
            [ L/2, -W/2],   # BR (near-right)
            [-L/2, -W/2],   # BL (near-left)
        ])
        H = cv2.getPerspectiveTransform(P, dst)

        # validate by checking local scale at the quadrilateral center
        c = P.mean(axis=0)
        ok = True
        try:
            base = np.array([[[c[0], c[1]]]], dtype=np.float32)
            px_x = np.array([[[c[0] + 1.0, c[1]]]], dtype=np.float32)
            px_y = np.array([[[c[0], c[1] + 1.0]]], dtype=np.float32)
            b   = cv2.perspectiveTransform(base, H)[0,0]
            bx1 = cv2.perspectiveTransform(px_x, H)[0,0]
            by1 = cv2.perspectiveTransform(px_y, H)[0,0]
            dx_m = float(bx1[0] - b[0])
            dy_m = float(by1[1] - b[1])
            if not np.isfinite(dx_m) or not np.isfinite(dy_m) or abs(dx_m) < 1e-4 or abs(dy_m) < 1e-4:
                ok = False
        except Exception:
            ok = False

        if ok:
            self.H = H
        else:
            # reject degenerate homography; render() will use safe fallbacks
            self.H = None

        
    
    def _project_px_to_court_m(self, xy_px):
        if self.H is None or xy_px is None:
            return None
        x, y = float(xy_px[0]), float(xy_px[1])
        pt = np.array([[ [x, y] ]], dtype=np.float32)  # shape (1,1,2)
        out = cv2.perspectiveTransform(pt, self.H)[0,0]  # (u,v) in meters
        return float(out[0]), float(out[1])
    
    def _kps_to_xyz(self, kps_xy, bbox=None):
        
        if kps_xy is None:
            return None
        K = kps_xy.shape[0]
        xyz = np.full((K,3), np.nan, dtype=np.float32)

        # Placeholder mapping: normalize by image size later (we’ll get size from caller)
        # We'll fill X,Y in caller; here we generate Z profile:
        # Raise shoulders/head slightly, keep ankles near z=0
        # Indices follow COCO-ish ordering. If your model differs, adjust.
        high = [0,1,2,3,4,5,6]   # head + shoulders set a bit higher
        mid  = [7,8,9,10]        # elbows / wrists
        low  = [11,12,13,14,15,16]  # hips / knees / ankles

        # base Z scale by bbox height if available
        zscale = 0.6
        if bbox is not None:
            h = max(1.0, bbox[3]-bbox[1])
            zscale = np.clip(h/400.0, 0.3, 1.2)

        for i in range(K):
            if np.isnan(kps_xy[i,0]) or np.isnan(kps_xy[i,1]):
                continue
            if i in high:
                z = 1.6 * zscale
            elif i in mid:
                z = 1.0 * zscale
            else:
                z = 0.1 * zscale
            xyz[i,2] = z
        return xyz
    
    
    
    def _outer_rect_edges_m(self):
        # outer singles rectangle in meters
        halfL = COURT_LEN / 2.0
        halfW = COURT_WID / 2.0
        return (-halfL, halfL, -halfW, halfW)  # (xmin,xmax,ymin,ymax)

    def _snap_to_outer_edge_m(self, x, y):
        xmin, xmax, ymin, ymax = self._outer_rect_edges_m()
        # distances to 4 edges
        d = [
            (abs(x - xmin), "left"),
            (abs(x - xmax), "right"),
            (abs(y - ymin), "bottom"),
            (abs(y - ymax), "top"),
        ]
        _, side = min(d, key=lambda t: t[0])
        if side == "left":   x = xmin
        if side == "right":  x = xmax
        if side == "bottom": y = ymin
        if side == "top":    y = ymax
        return x, y

    def _anchor_from_kps_px(self, kps_px, w, h):
        """Prefer ankles (15,16), else hips (11,12), else median of all valid."""
        import numpy as np
        idx_sets = [[15,16], [11,12]]
        pts = np.asarray([(float(p[0]), float(p[1])) for p in kps_px if p is not None], dtype=np.float32)
        if pts.size == 0:
            return None
        def pick(ids):
            got = []
            for i in ids:
                if i < len(kps_px) and kps_px[i] is not None:
                    got.append(kps_px[i])
            if got:
                g = np.asarray(got, dtype=np.float32)
                return float(g[:,0].mean()), float(g[:,1].mean())
            return None
        for ids in idx_sets:
            a = pick(ids)
            if a is not None:
                return a
        # fallback: median of all valid joints
        return float(np.median(pts[:,0])), float(np.median(pts[:,1]))

    def _shrink_xy_around_anchor(self, XY_m, ax, ay):
        
        import numpy as np
        X = XY_m[:, 0].copy()
        Y = XY_m[:, 1].copy()
        for i in range(len(X)):
            if np.isnan(X[i]) or np.isnan(Y[i]):
                continue
            dx = (X[i] - ax) * float(self.skel_scale)
            dy = (Y[i] - ay) * float(self.skel_scale)
            r = (dx*dx + dy*dy) ** 0.5
            if r > self.skel_max_r_m:
                s = self.skel_max_r_m / (r + 1e-6)
                dx *= s; dy *= s
            X[i] = ax + dx
            Y[i] = ay + dy
        return X, Y
    def _draw_skeleton(self, X, Y, Z, player_idx, color=(255, 255, 0)):
        
        # normalize color
        if max(color) <= 1.0:
            col = tuple(color)
        else:
            # FIX: ensure color interpreted as RGB, not BGR
            col = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

        lw = getattr(self, "skel_lw", 2.4)
        s  = getattr(self, "skel_pt", 18)

        # lines
        for a, b in SKELETON_PAIRS:
            if a < len(X) and b < len(X):
                if not (np.isnan(X[a]) or np.isnan(Y[a]) or np.isnan(Z[a]) or
                        np.isnan(X[b]) or np.isnan(Y[b]) or np.isnan(Z[b])):
                    ln, = self.ax.plot([X[a], X[b]], [Y[a], Y[b]], [Z[a], Z[b]],
                                    color=col, lw=lw)
                    self.player_lines[player_idx].append(ln)

        # joints
        for i in range(len(X)):
            if not (np.isnan(X[i]) or np.isnan(Y[i]) or np.isnan(Z[i])):
                pt = self.ax.scatter([X[i]], [Y[i]], [Z[i]], color=col, s=s, depthshade=False)
                self.player_joints[player_idx].append(pt)
    
    
    def _local_m_per_px(self, ax_px, ay_px):
       
        if self.H is None:
            return None
        import numpy as np, cv2
        base = np.array([[[ax_px, ay_px]]], dtype=np.float32)
        px_x = np.array([[[ax_px + 1.0, ay_px]]], dtype=np.float32)
        px_y = np.array([[[ax_px, ay_px + 1.0]]], dtype=np.float32)

        b   = cv2.perspectiveTransform(base, self.H)[0,0]
        bx1 = cv2.perspectiveTransform(px_x, self.H)[0,0]
        by1 = cv2.perspectiveTransform(px_y, self.H)[0,0]

        dx_m = float(bx1[0] - b[0])   # m per +1 px in image x
        dy_m = float(by1[1] - b[1])   # m per +1 px in image y

        # ❗ if either axis is ~0 or not finite, treat as invalid (prevents collapse)
        if not np.isfinite(dx_m) or not np.isfinite(dy_m) or abs(dx_m) < 1e-4 or abs(dy_m) < 1e-4:
            return None
        return dx_m, dy_m


    def render(self, frame_size, players_kps_px, players_bbox_px, ball_px=None):
        
        import cv2, numpy as np

        h, w = frame_size
        self._clear_dynamic()  # clear previous frame's scatters/lines

        halfL = COURT_LEN / 2.0
        halfW = COURT_WID / 2.0

        def _fallback_px_to_xy(px):
            X = (px[:, 0] / max(1.0, w)) * (2 * halfL) - halfL
            Y = (px[:, 1] / max(1.0, h)) * (2 * halfW) - halfW
            return X, Y

        def _proj_px_to_court_xy(px):
            if self.H is None or px is None or len(px) == 0:
                return _fallback_px_to_xy(px)
            pts = px.astype(np.float32).reshape(-1, 1, 2)
            uv = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)  # meters on court
            return uv[:, 0], uv[:, 1]

        # ---------- PLAYERS ----------
        if players_kps_px is None:  players_kps_px = [None, None]
        if players_bbox_px is None: players_bbox_px = [None, None]

        for i in range(2):
            kps = players_kps_px[i] if i < len(players_kps_px) else None
            box = players_bbox_px[i] if i < len(players_bbox_px) else None
            if kps is None:
                continue

            # collect visible joints
            # collect visible joints (px)
            pts = []
            for p in kps:
                if p is None: pts.append((np.nan, np.nan))
                else:         pts.append((float(p[0]), float(p[1])))
            pts = np.asarray(pts, dtype=np.float32)

            valid = ~np.isnan(pts[:, 0])
            if not valid.any():
                continue

            # --- choose anchor in px (ankles->hips->median) ---
            # 1) anchor from ankles→hips→median (image px)
            anchor_px = self._anchor_from_kps_px(kps, w, h)
            if anchor_px is None: 
                continue
            ax_px, ay_px = anchor_px

            # 2) anchor -> meters and snap to outer edge
            if self.H is None:
                halfL, halfW = COURT_LEN/2.0, COURT_WID/2.0
                ax_m = (ax_px / max(1.0, w)) * (2*halfL) - halfL
                ay_m = (ay_px / max(1.0, h)) * (2*halfW) - halfW
            else:
                pt = np.array([[[ax_px, ay_px]]], dtype=np.float32)
                uv = cv2.perspectiveTransform(pt, self.H)[0,0]
                ax_m, ay_m = float(uv[0]), float(uv[1])
                if self.uses_inner_box:
                    ax_m *= self._x_stretch     # stretch baseline direction
                    ay_m *= self._y_stretch     # stretch sideline direction
            if not getattr(self, "allow_outside", False):
                ax_m, ay_m = self._clamp_inside_court_m(ax_m, ay_m, margin=0.0)
            

            # 3) local meters-per-pixel at anchor (don’t be ~0!)
            mpp = self._local_m_per_px(ax_px, ay_px) if self.H is not None else None
            if mpp is None:   # fallback
                mx_per_px = COURT_LEN / max(1.0, w)
                my_per_px = COURT_WID / max(1.0, h)
            else:
                mx_per_px, my_per_px = mpp

            gain = getattr(self, "pose_xy_gain", 0.85)
            rmax = getattr(self, "pose_xy_clip_m", 1.10)

            # 4) per-joint XY around the anchor (PRESERVE SPREAD)
            K  = len(kps)
            Xm = np.full((K,), np.nan, np.float32)
            Ym = np.full((K,), np.nan, np.float32)
            for j, p in enumerate(kps):
                if p is None or np.isnan(p[0]) or np.isnan(p[1]):
                    continue
                dx_px = float(p[0]) - ax_px
                dy_px = float(p[1]) - ay_px
                xj = ax_m + gain * mx_per_px * dx_px
                yj = ay_m + gain * my_per_px * dy_px
                # clamp radius so figure stays compact
                rx, ry = xj - ax_m, yj - ay_m
                r = (rx*rx + ry*ry) ** 0.5
                if r > rmax:
                    s = rmax / (r + 1e-6)
                    xj = ax_m + rx*s;  yj = ay_m + ry*s
                Xm[j], Ym[j] = xj, yj

            # 5) Z profile (your existing helper), then plot coords
            Zonly = self._kps_to_xyz(kps, bbox=box)
            Zs = Zonly[:,2].astype(np.float32) if Zonly is not None else np.full((K,), 0.5, np.float32)
            Zs *= getattr(self, "skel_z_scale", 0.50)

            Xp = np.where(np.isnan(Xm), np.nan, self.scale_x * Xm + self.offset_xy[0])
            Yp = np.where(np.isnan(Ym), np.nan, self.scale_y * Ym + self.offset_xy[1])
            self._draw_skeleton(Xp, Yp, Zs, player_idx=i, color=(255,255,0))
            # --- 2D player marker ---
            s = getattr(self, "width_scale_2d", 1.0)
            p2d = self.ax2d.scatter([s*ay_m], [ax_m], s=90, color=(1, 1, 0), zorder=5)
            self._ax2d_dyn.append(p2d)








        # ---------- BALL ----------
        if ball_px is not None:
            bx_px, by_px = float(ball_px[0]), float(ball_px[1])

            if self.H is None:
                # proportional fallback
                bx_m = (bx_px / max(1.0, w)) * (2 * halfL) - halfL
                by_m = (by_px / max(1.0, h)) * (2 * halfW) - halfW
            else:
                pt = np.array([[[bx_px, by_px]]], dtype=np.float32)
                uv = cv2.perspectiveTransform(pt, self.H)[0, 0]
                bx_m, by_m = float(uv[0]), float(uv[1])
                if self.uses_inner_box:
                    bx_m *= self._x_stretch
                    by_m *= self._y_stretch
            if not getattr(self, "allow_outside", False):
                bx_m, by_m = self._clamp_inside_court_m(bx_m, by_m, margin=0.0)
            
            # --- 2D ball marker ---
            # --- 2D ball marker ---
            b2d = self.ax2d.scatter([s*by_m], [bx_m], s=80, color=(1,0,0), zorder=6)
            self._ax2d_dyn.append(b2d)
            # plot coords (same transform as court)
            Bx = self.scale_x * bx_m + self.offset_xy[0]
            By = self.scale_y * by_m + self.offset_xy[1]

            # --- vertical physics tied to ball XY motion ---
            dt = 1.0 / max(1.0, self.fps)

            # planar speed (used only to detect hits & give a small vz kick)
            if self._prev_ball_xy_m is None:
                v_planar = 0.0
            else:
                dx = bx_m - self._prev_ball_xy_m[0]
                dy = by_m - self._prev_ball_xy_m[1]
                v_planar = ((dx*dx + dy*dy) ** 0.5) / dt

            # Heuristic "hit" detector: sudden planar speed → give an upward kick
            # (keeps bounces realistic without needing 3D detection)
            if v_planar > 6.0 and self.ball_z <= 0.02 and self.ball_vz <= 0.2:
                self.ball_vz = min(8.0, self.hit_boost_k * v_planar + self.hit_boost_c)

            # Integrate vertical motion
            self.ball_vz -= self.g * dt
            self.ball_z  += self.ball_vz * dt

            # Bounce on ground
            if self.ball_z <= 0.0:
                # just hit the ground if we were descending
                if self.ball_vz < -self.min_bounce_vz:
                    self.ball_z  = 0.0
                    self.ball_vz = -self.bounce_e * self.ball_vz  # reflect with loss
                    # start a short 2D bounce flash
                    self._bounce_flash_t = 0.15
                else:
                    # settle on ground
                    self.ball_z = 0.0
                    self.ball_vz = 0.0

            # decay the flash timer
            if self._bounce_flash_t > 0.0:
                self._bounce_flash_t = max(0.0, self._bounce_flash_t - dt)

            self._prev_ball_xy_m = (bx_m, by_m)

            # --- draw 3D ball at physics height ---
            self.ball_scatter = self.ax.scatter([Bx], [By], [self._ball_alt],
                                    color=(0, 1, 0), s=26, depthshade=False)

        # ---------- RENDER ----------
        self.canvas.draw()
        buf = np.asarray(self.canvas.buffer_rgba())
        rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        return rgb

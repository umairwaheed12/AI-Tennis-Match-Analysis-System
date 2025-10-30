# src/lifter3d.py
# Minimal interface wrapper so you can swap in any 3D lifter you prefer.
# Expect COCO-17 order in 2D (same order you already draw), returns meters.

import numpy as np

class PoseLifter3D:
    def __init__(self, backend="dummy", person_height_m=1.75):
        self.backend = backend
        self.person_h = float(person_height_m)
        # TODO: load your actual model here (VideoPose3D / MMPose), keep a .predict_3d() API.

    def _dummy_lift(self, k2d):
        
        K = k2d.shape[0]
        J = np.zeros((K, 3), np.float32)
        # normalize around pelvis (avg hips)
        hips = []
        for idx in [11, 12]:
            if not np.isnan(k2d[idx,0]):
                hips.append(k2d[idx])
        c = np.nanmean(np.stack(hips), axis=0) if hips else np.nanmean(k2d, axis=0)
        K2 = k2d - c  # center
        # simple scale from person height in pixels -> meters
        valid_y = K2[~np.isnan(K2[:,1]),1]
        pix_h = (np.nanmax(valid_y) - np.nanmin(valid_y)) if valid_y.size else 300.0
        s = (self.person_h / max(1e-3, pix_h))
        # XY from 2D, Z from heuristic profile (shoulders forward, legs back)
        forward = {5,6,7,8,9,10}
        back    = {13,14,15,16}
        for i in range(K):
            if np.isnan(k2d[i,0]): continue
            J[i,0] =  s * K2[i,0]
            J[i,1] = -s * K2[i,1]  # image y down -> +up in panel coords
            z = 0.0
            if i in forward: z += 0.12
            if i in back:    z -= 0.08
            J[i,2] = z
        # scale to meters roughly; pelvis near (0,0,0)
        return J

    def predict(self, k2d_xy: np.ndarray) -> np.ndarray:
        
        if self.backend == "dummy":
            return self._dummy_lift(k2d_xy)
        # else: call your real model here and return meters
        raise NotImplementedError("Hook your 3D lifter backend here.")

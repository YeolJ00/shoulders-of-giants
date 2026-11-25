"""Episodes built by hand for evaluating a trained model on situations given as numbers (a real
experiment's geometry and masses) rather than recorded by the teacher. They expose the same
attributes as lsim.train.windows.Episode, so make_window and rollout run on them unchanged."""
from types import SimpleNamespace

import numpy as np

from lsim.data.episode import ramp_pose
from lsim.data.scene import PARK
from lsim.train.windows import Body, Episode, _boxes_points, gripper_mesh


class SyntheticEpisode(Episode):
    def __init__(self, bodies, q, mass, com, inertia, wd, K=3):
        self.bodies, self.q = bodies, q
        self.qd = np.zeros(q.shape[:2] + (6,), np.float32)
        self.mass, self.com, self.inertia = mass, com, inertia
        self.wd, self.w, self.K, self.meta = wd, 0, K, {"K": K}
        self.idx = np.zeros((0, 4), np.int64)
        self.F = self.P = np.zeros((0, 3), np.float32)
        self.g_idx = np.zeros((0, 2), np.int64)
        self.g_F = self.g_P = np.zeros((0, 3), np.float32)
        self.pokes = []


def ramp_cube(edge, mass, mu, angle_deg, s0, frames, restitution=0.3, ramp_mu=None):
    """A solid cube (edge m, mass kg, friction mu) at rest on a plank tilted angle_deg, s0 m up
    the slope, with the gripper parked; `frames` frames of identical poses to roll out from."""
    th = float(np.radians(angle_deg))
    ramp = {"angle": th, "yaw": 0.0, "low": [-0.25, 0.0], "mid": [-0.25 + 0.25 * np.cos(th), 0.0],
            "mu": float(mu if ramp_mu is None else ramp_mu), "restitution": 0.3}
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * edge / 2
    pos, quat = ramp_pose(SimpleNamespace(ramp=ramp), np.array([0, 0, 0, 1.0]), corners, s0)
    unit_I = np.full(3, edge * edge / 6.0)
    none = np.zeros((0, 2), np.int32)
    cube = Body("cube", "dynamic", "rigid", 0, _boxes_points([((0, 0, 0), (edge / 2,) * 3)], 0.015), none, 2,
                {"mass": mass, "mu": mu, "restitution": restitution, "com_offset": [0.0, 0.0, 0.0],
                 "inertia_scale": [1.0, 1.0, 1.0], "solid": 1.0, "size": edge}, (unit_I, np.eye(3)))
    grip = Body("gripper", "kinematic", "rigid", 1, gripper_mesh(), none, 1)
    q = np.zeros((frames + 1, 2, 7), np.float32)
    q[:, 0, :3], q[:, 0, 3:] = pos, quat
    q[:, 1, :3], q[:, 1, 6] = PARK, 1.0
    mass_a = np.array([mass, 0.5], np.float32)
    com = np.zeros((2, 3), np.float32)
    inertia = np.stack([np.diag(unit_I * mass), np.eye(3) * 1e-3]).astype(np.float32)
    wd = {"objects": [{"name": "cube", "mass": mass}], "pokes": [], "hinge": None, "ramp": ramp,
          "spec": [{"on_ramp": s0}], "families_sampled": []}
    return SyntheticEpisode([cube, grip], q, mass_a, com, inertia, wd), ramp

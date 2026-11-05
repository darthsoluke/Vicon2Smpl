"""Outputs for downstream use: an AMASS-style .npz recorder and a UDP stream.

UDP packet (little endian), one per solved frame:
    char[4]  magic  b"VSMP"
    uint32   version (1)
    int64    vicon frame number
    float64  host timestamp (s, time.perf_counter of the frame's arrival)
    uint32   num_joints J
    float32  rmse (m)
    float32[3]    trans (m, Y-up)
    float32[J*3]  pose, axis-angle per joint (smplx order; pose[0:3] = global_orient)
"""
import socket
import struct
import numpy as np

HEADER = struct.Struct("<4sIqdIf")


class UdpPublisher:
    def __init__(self, target):
        host, port = target.rsplit(":", 1)
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, frame_no, t, pose_aa, trans, rmse):
        J = len(pose_aa)
        payload = HEADER.pack(b"VSMP", 1, int(frame_no), float(t), J, float(rmse if np.isfinite(rmse) else -1)) \
            + np.asarray(trans, "<f4").tobytes() + np.asarray(pose_aa, "<f4").reshape(-1).tobytes()
        self.sock.sendto(payload, self.addr)

    @staticmethod
    def decode(data):
        magic, ver, frame, t, J, rmse = HEADER.unpack_from(data)
        assert magic == b"VSMP"
        arr = np.frombuffer(data, "<f4", offset=HEADER.size)
        return {"frame": frame, "time": t, "rmse": rmse, "trans": arr[:3], "pose": arr[3:].reshape(J, 3)}


class SequenceRecorder:
    """Collects solved frames and saves an AMASS-like npz (poses / trans / betas / mocap_framerate)."""

    def __init__(self, model_type, gender, betas, fps, model=None):
        """With `model`, `poses`/`trans` are stored in the model's published convention (SOMA: SOMALayer.pose
        inputs) and the solver's own parameters are kept as `solver_poses`/`solver_trans`."""
        self.meta = dict(model_type=model_type, gender=gender, betas=np.asarray(betas), mocap_framerate=float(fps or 0))
        # only models with a published-convention hook convert (SOMA); others record solver parameters as before
        self.model = model if hasattr(model, "public_pose") else None
        if self.model is not None:
            self.meta.update(self.model.recording_metadata(betas))
        self.poses, self.trans, self.frames, self.rmse = [], [], [], []
        self.solver_poses, self.solver_trans = [], []

    def add(self, frame_no, pose_aa, trans, rmse):
        if self.model is not None:
            self.solver_poses.append(np.asarray(pose_aa, np.float32).reshape(-1))
            self.solver_trans.append(np.asarray(trans, np.float32))
            pose_aa, trans = self.model.public_pose(pose_aa, trans)
        self.poses.append(np.asarray(pose_aa, np.float32).reshape(-1))
        self.trans.append(np.asarray(trans, np.float32))
        self.frames.append(frame_no)
        self.rmse.append(rmse)

    def save(self, path):
        poses = np.array(self.poses)
        extra = {}
        if self.solver_poses:
            extra = dict(solver_poses=np.array(self.solver_poses), solver_trans=np.array(self.solver_trans))
        np.savez(path, poses=poses, root_orient=poses[:, :3], trans=np.array(self.trans), frame_numbers=np.array(self.frames), rmse=np.array(self.rmse),
                 **self.meta, **extra)
        return len(poses)

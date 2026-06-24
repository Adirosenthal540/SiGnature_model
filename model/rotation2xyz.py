# This code is based on https://github.com/Mathux/ACTOR.git
import torch
import utils.rotation_conversions as geometry
import smplx
import time

# from .get_model import JOINTSTYPES
JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]


class Rotation2xyz:
    def __init__(self, device):
        self.device = device
        self.smplx_model = None

    def load_and_freeze_smplx(self, data):
        if self.smplx_model is None:
            smplx_model = (
                smplx.create(
                    data.args.data_path_1 + "smplx_models/",
                    model_type="smplx",
                    gender="NEUTRAL_2020",
                    use_face_contour=False,
                    num_betas=0,
                    ext="npz",
                    use_pca=False,
                    num_expression_coeffs=0,
                    use_face=False,
                )
                .to(self.device)
                .eval()
            )

            # Freeze smplx weights
            for p in smplx_model.parameters():
                p.requires_grad = False
            self.smplx_model = smplx_model

        return self.smplx_model

    def __call__(self, x, pose_rep, betas=None, expression=None, glob_rot=None, data=None, use_global=True, **kwargs):
        if pose_rep == "xyz":
            return x

        self.smplx = self.load_and_freeze_smplx(data)
        bs, n = x.shape[0], x.shape[-1]
        n_joints = 55
        # Inverse normalization
        sample = x.squeeze(dim=2).permute(0, 2, 1).to(self.device)  # [337, n_frames] -> [n_frames, 337]
        sample = data.inv_transform(sample)

        # Split the network output to the different parts
        rotations_6d = sample[:, :, : n_joints * 6]
        if use_global:
            translations = sample[:, :, n_joints * 6 : n_joints * 6 + 3]
        else:
            translations = torch.zeros((bs * n, 3)).to(self.device)

        # Convert rot6d to pose 3d
        n_frames: int = rotations_6d.shape[1]
        rotations_matrix = geometry.rotation_6d_to_matrix(rotations_6d.reshape(bs, n_frames, n_joints, 6))
        rotations_angle = geometry.matrix_to_axis_angle(rotations_matrix).reshape(bs, n_frames, n_joints * 3)

        rec_pose = rotations_angle.reshape(bs * n, -1).to(self.device)
        rec_trans = translations.reshape(bs * n, -1).to(self.device)

        if betas is None:
            betas = torch.zeros((bs * n, 0))
        if expression is None:
            expression = torch.zeros((bs * n, 0))

        with torch.no_grad():
            smplx_output_rec = self.smplx(
                transl=rec_trans,
                betas=betas.to(self.device),
                expression=expression.to(self.device),
                jaw_pose=rec_pose[:, 66:69],
                global_orient=rec_pose[:, :3],
                body_pose=rec_pose[:, 3 : 21 * 3 + 3],
                left_hand_pose=rec_pose[:, 25 * 3 : 40 * 3],
                right_hand_pose=rec_pose[:, 40 * 3 : 55 * 3],
                return_joints=True,
                leye_pose=rec_pose[:, 69:72],
                reye_pose=rec_pose[:, 72:75],
                return_verts=False,
            )

        return smplx_output_rec.joints[:, :n_joints].reshape(bs, n, -1, 3)


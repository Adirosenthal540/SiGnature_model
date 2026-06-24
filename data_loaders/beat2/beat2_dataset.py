import os
import numpy as np
import lmdb as lmdb
import torch
import pickle
from torch.utils.data import Dataset
import smplx
import pyarrow as pa
import json

from .utils import config
from data_loaders.beat2.utils import rotation_conversions as rc
from data_loaders.beat2.utils.build_vocab import Vocab
from data_loaders.beat2.utils.cache_utils import Beat2CacheGenerator
from data_loaders.beat2.data_tools import joints_list

from data_loaders.beat2.utils.cache_utils import calculate_mean_std


def pa_serialize(obj) -> pa.Buffer:
    """Mimic `pyarrow.serialize(obj).to_buffer()`."""
    return pa.py_buffer(pickle.dumps(obj, protocol=5))  # protocol 5 = Py3.8+


def pa_deserialize(buf: pa.Buffer):
    """Mimic `pyarrow.deserialize(buf)`."""
    # `memoryview` keeps it zero-copy if `buf` already points to shared memory
    return pickle.loads(memoryview(buf))


class _VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Vocab":
            return Vocab
        return super().find_class(module, name)


def _load_vocab(f):
    return _VocabUnpickler(f).load()


class BEAT2Dataset(Dataset):
    """Beat 2 data set object."""

    def __init__(self, split: str, build_cache: bool = True, device=0, use_amass=False, use_seg=False, cache_path=None):
        args = config.parse_args()
        if cache_path is not None:
            args.cache_path = cache_path
            args.training_speakers = [int(cache_path.split("_")[-1])]
        else:
            args.cache_path = args.cache_path

        self.args = args
        self.loaded_args: dict = None

        # Change in the future to support multiple processes
        self.device = device

        if args.word_rep is not None:
            with open(os.path.join(args.data_path, "weights/vocab.pkl"), "rb") as f:
                self.lang_model = _load_vocab(f)

        # Build cache
        if build_cache:
            self.cache_gnerator = Beat2CacheGenerator(args, device, self.lang_model, use_amass=use_amass, use_seg=use_seg)
            self.smplx = self.cache_gnerator.get_smplx()
            for split_ in ["train", "val", "test"]:
                cache_folder_path = os.path.join(self.args.root_path, self.args.cache_path, split_, f"{args.pose_rep}_cache")
                self.cache_gnerator.build_cache(cache_folder_path, data_folder=args.data_path, split=split_, force_build=self.args.new_cache)
            # Calculate mean and the std values of dataset
            calculate_mean_std(args.cache_path)
            # calculate_mean_std(args.amass_path)
        cache_folder_path = os.path.join(self.args.root_path, self.args.cache_path, split, f"{args.pose_rep}_cache")

        # Load cache
        self.lmdb_env = lmdb.open(cache_folder_path, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]

        # Load args
        with open(os.path.join(cache_folder_path, "args.json"), "r") as f:
            self.loaded_args = json.load(f)
            self.selected_files = self.loaded_args["selected_files"]

        # Load mean and the std values of dataset
        if self.args.beat_align:
            cache_root_folder_path: str = os.path.join(args.root_path, args.cache_path)
            if not os.path.exists(os.path.join(cache_root_folder_path, "Mean.npy")) or not os.path.exists(
                os.path.join(cache_root_folder_path, "Std.npy")
            ):
                raise ValueError("Align argument is set but no mean and std values to cache data")

            # Load mean and std values
            self.mean = np.load(os.path.join(cache_root_folder_path, "Mean.npy"))
            self.std = np.load(os.path.join(cache_root_folder_path, "Std.npy"))

            if not os.path.exists(args.data_path + f"weights/mean_vel_{args.pose_rep}.npy"):
                self._calculate_mean_velocity(args.data_path + f"weights/mean_vel_{args.pose_rep}.npy")
            self.avg_vel = np.load(args.data_path + f"weights/mean_vel_{args.pose_rep}.npy")
            # self.avg_vel_rot = np.load(args.data_path + f"weights/mean_vel_{args.pose_rep}_rot.npy")
            # Load velocity mean
            # if not os.path.exists(os.path.join(cache_root_folder_path, "avg_vel.npy")):
            #     raise ValueError("Align argument is set but no mean and std values to cache data")

            # self.avg_vel = np.load(os.path.join(cache_root_folder_path, "avg_vel.npy"))

        # Get joints order and joints mask
        self.ori_joint_list: dict = joints_list[self.args.ori_joints]
        self.tar_joint_list: dict = joints_list[self.args.tar_joints]
        if "smplx" in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
            self.joints = len(list(self.tar_joint_list.keys()))
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys())) + 1
            self.joint_mask = np.zeros(self.joints * 3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][1]] = 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pa_deserialize(sample)
            # tar_pose, in_audio, in_audio_resample, in_facial, in_shape, in_word, emo, sem, vid, trans = sample
            tar_pose, in_audio, in_audio_resample, in_facial, in_shape, in_word, emo, sem, vid, name, text_semantic, trans = sample
            emo = torch.from_numpy(emo).int()
            sem = torch.from_numpy(sem).float()
            in_audio = torch.from_numpy(in_audio).float()
            in_audio_resample = torch.from_numpy(in_audio_resample).float()
            in_word = torch.from_numpy(in_word).float() if self.args.word_cache else torch.from_numpy(in_word).int()
            tar_pose = torch.from_numpy(tar_pose).float()
            trans = torch.from_numpy(trans).float()
            in_facial = torch.from_numpy(in_facial).float()
            vid = torch.from_numpy(vid).float()
            in_shape = torch.from_numpy(in_shape).float()
            if text_semantic is None:
                text_semantic = ""

            in_text = self._create_text_from_in_word(in_word)
            dict_data = {
                "pose": tar_pose,
                "audio": in_audio,
                "audio_resample": in_audio_resample,
                "facial": in_facial,
                "beta": in_shape,
                "word": in_word,
                "id": vid,
                "name": name,
                "emo": emo,
                "sem": sem,
                "trans": trans,
                "text_semantic": text_semantic,
                "text": in_text,
            }
            dict_data_processed = self._load_data(dict_data)
            return dict_data_processed

    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :165]
        tar_contact = tar_pose_raw[:, 165:169]
        tar_trans = dict_data["trans"]
        tar_exps = dict_data["facial"]
        in_audio = dict_data["audio"]
        in_audio_resample = dict_data["audio_resample"]
        in_word = dict_data["word"]
        tar_beta = dict_data["beta"]
        tar_id = dict_data["id"]
        tar_name = dict_data["name"]
        in_text = dict_data["text"]
        in_text_semantic = dict_data["text_semantic"]
        n, j = tar_pose.shape[0], self.joints

        tar_pose_matrix = rc.axis_angle_to_matrix(tar_pose.reshape(n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_matrix).reshape(n, 55 * 6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)

        # Z Normalization
        if self.args.beat_align:
            latent_all[:, :333] = (latent_all[:, :333] - self.mean[:333]) / self.std[:333]

        return {
            "in_audio": in_audio,
            "in_audio_resample": in_audio_resample,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "tar_id": tar_id,
            "tar_name": tar_name,
            "latent_all": latent_all,
            "tar_pose_6d": tar_pose_6d,
            "tar_contact": tar_contact,
            "in_text_semantic": in_text_semantic,
            "text": in_text,
        }

    def _calculate_mean_velocity(self, save_path):
        dir_p = self.args.data_path + self.args.pose_rep + "/"
        all_list_xyz = []
        all_list_rot = []
        from tqdm import tqdm

        for tar in tqdm(os.listdir(dir_p)):
            if tar.endswith(".npz"):
                m_data = np.load(dir_p + tar, allow_pickle=True)
                betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
                exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
                trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
                max_length = 1000
                s, r = n // max_length, n % max_length
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[i * max_length : (i + 1) * max_length],
                            transl=trans[i * max_length : (i + 1) * max_length],
                            expression=exps[i * max_length : (i + 1) * max_length],
                            jaw_pose=poses[i * max_length : (i + 1) * max_length, 66:69],
                            global_orient=poses[i * max_length : (i + 1) * max_length, :3],
                            body_pose=poses[i * max_length : (i + 1) * max_length, 3 : 21 * 3 + 3],
                            left_hand_pose=poses[i * max_length : (i + 1) * max_length, 25 * 3 : 40 * 3],
                            right_hand_pose=poses[i * max_length : (i + 1) * max_length, 40 * 3 : 55 * 3],
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[i * max_length : (i + 1) * max_length, 69:72],
                            reye_pose=poses[i * max_length : (i + 1) * max_length, 72:75],
                        )["joints"][:, :55, :].reshape(max_length, 55 * 3)
                    all_tensor.append(joints)
                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s * max_length : s * max_length + r],
                            transl=trans[s * max_length : s * max_length + r],
                            expression=exps[s * max_length : s * max_length + r],
                            jaw_pose=poses[s * max_length : s * max_length + r, 66:69],
                            global_orient=poses[s * max_length : s * max_length + r, :3],
                            body_pose=poses[s * max_length : s * max_length + r, 3 : 21 * 3 + 3],
                            left_hand_pose=poses[s * max_length : s * max_length + r, 25 * 3 : 40 * 3],
                            right_hand_pose=poses[s * max_length : s * max_length + r, 40 * 3 : 55 * 3],
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[s * max_length : s * max_length + r, 69:72],
                            reye_pose=poses[s * max_length : s * max_length + r, 72:75],
                        )["joints"][:, :55, :].reshape(r, 55 * 3)
                    all_tensor.append(joints)
                joints = torch.cat(all_tensor, axis=0)
                joints = joints.permute(1, 0)
                dt = 1 / 30
                # first steps is forward diff (t+1 - t) / dt
                init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
                # middle steps are second order (t+1 - t-1) / 2dt
                middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
                # last step is backward diff (t - t-1) / dt
                final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
                vel_seq_xyz = torch.cat([init_vel, middle_vel, final_vel], dim=1).permute(1, 0).reshape(n, 55, 3)
                # .permute(1, 0).reshape(n, 55, 3)
                vel_seq_xyz_np = vel_seq_xyz.cpu().numpy()
                vel_joints_xyz_np = np.linalg.norm(vel_seq_xyz_np, axis=2)  # n * 55

                all_list_xyz.append(vel_joints_xyz_np)

                poses_rot = poses.permute(1, 0)
                # first steps is forward diff (t+1 - t) / dt
                init_vel = (poses_rot[:, 1:2] - poses_rot[:, :1]) / dt
                # middle steps are second order (t+1 - t-1) / 2dt
                middle_vel = (poses_rot[:, 2:] - poses_rot[:, 0:-2]) / (2 * dt)
                # last step is backward diff (t - t-1) / dt
                final_vel = (poses_rot[:, -1:] - poses_rot[:, -2:-1]) / dt
                vel_seq_rot = torch.cat([init_vel, middle_vel, final_vel], dim=1).permute(1, 0).reshape(n, 55, 3)
                # .permute(1, 0).reshape(n, 55, 3)
                vel_seq_rot_np = torch.abs(vel_seq_rot.cpu().numpy())
                # vel_joints_rot_np = np.linalg.norm(vel_seq_rot_np, axis=2)  # n * 55

                all_list_rot.append(vel_seq_rot_np)
        avg_vel_xyz = np.mean(np.concatenate(all_list_xyz, axis=0), axis=0)  # 55
        avg_vel_rot = np.mean(np.concatenate(all_list_rot, axis=0), axis=0)  # 55
        np.save(save_path, avg_vel_xyz)
        # np.save(save_path.replace(".npy", "_rot.npy"), avg_vel_rot)

    def _create_text_from_in_word(self, in_word):
        def remove_consecutive_duplicates(arr):

            arr = np.array(arr)

            # Initialize the new list with the first element of the original array
            new_arr = [arr[0]]

            # Iterate over the array from the second element to the end
            for i in range(1, len(arr)):
                # If the current element is not the same as the last element in the new list, append it
                if arr[i] != arr[i - 1]:
                    new_arr.append(arr[i])

            return new_arr

        new_in_word = remove_consecutive_duplicates(in_word)
        text = ""
        for t in new_in_word:
            text = f"{text} {self.lang_model.index2word[t]}"

        def clean_text(s):
            # First, replace two or more spaces with a single space
            import re

            s = re.sub(r"\s{2,}", " ", s)
            # Then, remove leading spaces
            s = s.lstrip()
            return s

        return clean_text(text)

    # def inv_transform(self, data, device=None):
    #     """The inverse transform of normalization"""
    #     if self.args.beat_align:
    #         if device is None:
    #             return data * self.std + self.mean
    #         else:
    #             return data * torch.from_numpy(self.std).to(device) + torch.from_numpy(self.mean).to(device)
    #     else:
    #         return data
    def inv_transform(self, data):
        """The inverse transform of normalization"""
        if self.args.beat_align:
            return data.to(self.device) * torch.from_numpy(self.std).to(self.device) + torch.from_numpy(self.mean).to(self.device)
        else:
            return data

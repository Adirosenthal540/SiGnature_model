import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist

# import pyarrow
import pickle
import librosa
import smplx
import glob
import random
from data_loaders.beat2.amass_generator import AMASS2CacheGenerator
from data_loaders.beat2.utils.build_vocab import Vocab
from data_loaders.beat2.utils.cache_utils import calculate_mean_std

# from .utils.audio_features import Wav2Vec2Model
from transformers import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools
from .utils import config

import codecs as cs
from tqdm import tqdm
from os.path import join as pjoin
import pyarrow as pa


def pa_serialize(obj) -> pa.Buffer:
    """Mimic `pyarrow.serialize(obj).to_buffer()`."""
    return pa.py_buffer(pickle.dumps(obj, protocol=5))  # protocol 5 = Py3.8+


def pa_deserialize(buf: pa.Buffer):
    """Mimic `pyarrow.deserialize(buf)`."""
    # `memoryview` keeps it zero-copy if `buf` already points to shared memory
    return pickle.loads(memoryview(buf))


class AMASSDataset(Dataset):
    def __init__(self, split: str, build_cache: bool = True, device=0):

        args = config.parse_args()
        self.args = args
        self.loaded_args: dict = None
        print("Loading dataset %s ..." % args.trainer)

        # Change in the future to support multiple processes
        self.device = device

        # Build cache
        if build_cache:
            self.cache_gnerator = AMASS2CacheGenerator(args, device)
            self.smplx = self.cache_gnerator.get_smplx()
            for split_ in ["train", "val", "test"]:
                cache_folder_path = os.path.join(self.args.root_path, self.args.cache_path + "_amass2", split_, f"{args.pose_rep}_cache")
                self.cache_gnerator.build_cache(cache_folder_path, data_folder=args.data_amass_path, split=split_, force_build=self.args.new_cache)
            # Calculate mean and the std values of dataset
            calculate_mean_std(args.cache_path + "_amass2")
            # calculate_mean_std(args.amass_path)
        cache_folder_path = os.path.join(self.args.root_path, self.args.cache_path + "_amass2", split, f"{args.pose_rep}_cache")

        # Load cache
        self.lmdb_env = lmdb.open(cache_folder_path, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]

        # Load args
        # with open(os.path.join(cache_folder_path, "args.json"), "r") as f:
        #     self.loaded_args = json.load(f)
        #     self.selected_files = self.loaded_args["selected_files"]

        # Load mean and the std values of dataset
        if self.args.beat_align:
            cache_root_folder_path: str = os.path.join(args.root_path, args.cache_path + "_amass2")
            if not os.path.exists(os.path.join(cache_root_folder_path, "Mean.npy")) or not os.path.exists(
                os.path.join(cache_root_folder_path, "Std.npy")
            ):
                raise ValueError("Align argument is set but no mean and std values to cache data")

            # Load mean and std values
            self.mean = np.load(os.path.join(cache_root_folder_path, "Mean.npy"))
            self.std = np.load(os.path.join(cache_root_folder_path, "Std.npy"))
            self.std[self.std == 0] = 1

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
            tar_pose, in_audio, in_audio_resample, in_facial, in_shape, in_word, emo, sem, vid, name, text_semantic, trans = sample
            emo = torch.from_numpy(np.array([-1])).int()
            sem = torch.from_numpy(np.array([-1])).float()
            in_audio = torch.from_numpy(np.array([-1])).float()
            in_audio = torch.zeros([104533, 2])
            in_audio_resample = torch.from_numpy(np.array([-1])).float()
            in_word = torch.zeros([196]).int()  # torch.from_numpy(np.array([-1])).float()
            tar_pose = torch.from_numpy(tar_pose).float()
            trans = torch.from_numpy(trans).float()
            in_facial = torch.from_numpy(np.array([-1])).float()
            vid = torch.from_numpy(vid).float()
            in_shape = torch.from_numpy(in_shape).float()
            if text_semantic is None:
                text_semantic = ""
            in_text = ""
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
            latent_all = (latent_all - self.mean) / self.std

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

    def inv_transform(self, data):
        """The inverse transform of normalization"""
        if self.args.beat_align:
            return data.to(self.device) * torch.from_numpy(self.std).to(self.device) + torch.from_numpy(self.mean).to(self.device)
        else:
            return data

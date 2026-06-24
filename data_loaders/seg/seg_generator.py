import json
import math
import textgrid as tg
from loguru import logger
import os
import lmdb
import librosa
from termcolor import colored
from collections import defaultdict
import numpy as np
import torch
from data_loaders.beat2.utils import rotation_conversions as rc
from data_loaders.beat2.data_tools import joints_list
import pandas as pd
import shutil
import smplx
import pickle
import pyarrow as pa
from tqdm import tqdm
from os.path import join as pjoin
from data_loaders.beat2.utils.config import parse_args
import glob
import torch.nn.functional as F


def pa_serialize(obj) -> pa.Buffer:
    """Mimic `pyarrow.serialize(obj).to_buffer()`."""
    return pa.py_buffer(pickle.dumps(obj, protocol=5))  # protocol 5 = Py3.8+


def pa_deserialize(buf: pa.Buffer):
    """Mimic `pyarrow.deserialize(buf)`."""
    # `memoryview` keeps it zero-copy if `buf` already points to shared memory
    return pickle.loads(memoryview(buf))


class SEG2CacheGenerator:
    """Create cache for beat2 dataset."""

    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device
        self.smplx = None
        self.args.audio_rep = None

    def build_cache(self, cache_folder_path: str, data_folder: str, split: str, force_build: bool = False):
        build_cache = False
        if os.path.exists(cache_folder_path):
            logger.info(f"Cache was found {format(cache_folder_path)}")
            if force_build:
                build_cache = True
                logger.info(f"Override cache")
                if os.path.exists(cache_folder_path):
                    shutil.rmtree(cache_folder_path)
        else:
            logger.info(f"Cache was not found {format(cache_folder_path)}")
            build_cache = True

        if build_cache:
            logger.info("Creating the dataset cache...")
            logger.info("Reading data '{}'...".format(data_folder))

            # # Get selected files
            # self._set_selected_files(split)

            self._cache_generation(
                cache_folder_path,
                split,
                data_folder,
                self.args.disable_filtering,
                self.args.clean_first_seconds,
                self.args.clean_final_seconds,
                is_test=False,
            )

            # Save args that created database
            logger.info(f"Saving arguments to file")
            self._save_args(cache_folder_path, data_folder)

    def get_smplx(self):
        # Create smplx model to create data
        if self.smplx is None:
            smplx_params = {
                "model_path": self.args.data_path_1 + "smplx_models/",
                "model_type": "smplx",
                "gender": "NEUTRAL_2020",
                "use_face_contour": False,
                "num_betas": 300,
                "num_expression_coeffs": 100,
                "ext": "npz",
                "use_pca": False,
            }

            self.smplx = smplx.create(**smplx_params).to(self.device).eval()
            self.smplx_params = smplx_params

        return self.smplx

    def _save_args(self, cache_folder_path: str, data_folder: str) -> None:
        args_dict = vars(self.args)
        args_dict["data_folder"] = data_folder
        # args_dict["selected_files"] = list(self.selected_files["id"])
        args_dict["smplx_params"] = self.smplx_params
        with open(os.path.join(cache_folder_path, "args.json"), "w") as fw:
            json.dump(args_dict, fw, indent=4, sort_keys=True)

    def _cache_generation(
        self,
        out_lmdb_dir,
        split,
        data_folder,
        disable_filtering,
        clean_first_seconds,
        clean_final_seconds,
        is_test=False,
    ):
        n_filtered_out = defaultdict(int)
        self.n_out_samples = 0
        self.ori_stride: int = self.args.test_stride if is_test else self.args.stride
        self.ori_length: int = self.args.pose_length
        self.alignment = [0, 0]  # for trinity

        self.pose_fps: int = self.args.pose_fps
        self.audio_sr: int = self.args.audio_sr

        self.ori_joint_list: dict = joints_list[self.args.ori_joints]
        self.tar_joint_list: dict = joints_list[self.args.tar_joints]

        if is_test:
            self.args.multi_length_training = [1.0]
        self.max_length = int(self.args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(self.args.pose_length / self.args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.test_length * self.args.audio_sr:
            self.max_audio_pre_len = self.args.test_length * self.args.audio_sr

        # Get joints order and joints mask
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

        # create db for samples
        if not os.path.exists(out_lmdb_dir):
            os.makedirs(out_lmdb_dir)
        if len(self.args.training_speakers) == 1:
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=int(1024**3 * 50))  # 50G
        else:
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=int(1024**3 * 200))  # 200G

        search_path = self.args.data_seg_path
        npz_files = glob.glob(os.path.join(search_path, "*.npz"), recursive=True)
        for index, file_name in enumerate(npz_files):
            f_name = file_name.split("/")[-1]
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_file = file_name
            pose_each_file = []
            trans_each_file = []
            trans_v_each_file = []
            shape_each_file = []
            audio_each_file = []
            audio_each_file_resample = []
            facial_each_file = []
            word_each_file = []
            emo_each_file = []
            sem_each_file = []
            vid_each_file = []
            id_pose = f_name  # 1_wayne_0_1_1

            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            if "smplx" in self.args.pose_rep:
                pose_data = np.load(pose_file, allow_pickle=True)
                if len(pose_data.files) == 6:
                    logger.info(colored(f"# ---- state file ---- #", "red"))
                    continue
                assert 30 % self.args.pose_fps == 0, "pose_fps should be an aliquot part of 30"
                # pose_each_file = self.load_amass(pose_data)
                fps = pose_data["mocap_frame_rate"]
                stride = round(fps / 30)
                pose_each_file = pose_data["poses"][::stride] * self.joint_mask
                length_pose = 90
                # pose_interpolate = F.interpolate(
                #     pose_each_file.permute(1, 0),
                #     size=length_pose,
                #     mode="linear",
                #     align_corners=True,
                # ).permute(1, 0)
                from scipy.interpolate import interp1d

                T, D = pose_each_file.shape
                old_idx = np.arange(T)
                new_idx = np.linspace(0, T - 1, length_pose)
                f = interp1d(old_idx, pose_each_file, axis=0, kind="linear")
                pose_interpolate = f(new_idx)
                pose_each_file = pose_interpolate[:, self.joint_mask.astype(bool)]

                trans_each_file = pose_data["trans"][::stride]
                f = interp1d(old_idx, trans_each_file, axis=0, kind="linear")
                trans_interpolate = f(new_idx)
                trans_each_file = trans_interpolate

                trans_each_file[:, 0] = trans_each_file[:, 0] - trans_each_file[0, 0]
                trans_each_file[:, 2] = trans_each_file[:, 2] - trans_each_file[0, 2]
                trans_v_each_file = np.zeros_like(trans_each_file)
                trans_v_each_file[1:, 0] = trans_each_file[1:, 0] - trans_each_file[:-1, 0]
                trans_v_each_file[0, 0] = trans_v_each_file[1, 0]
                trans_v_each_file[1:, 2] = trans_each_file[1:, 2] - trans_each_file[:-1, 2]
                trans_v_each_file[0, 2] = trans_v_each_file[1, 2]
                trans_v_each_file[:, 1] = trans_each_file[:, 1]

                shape_each_file = np.repeat(pose_data["betas"].reshape(1, -1), pose_each_file.shape[0], axis=0)

            if self.args.id_rep is not None:
                vid_each_file = np.repeat(np.array(int(-1)).reshape(1, 1), pose_each_file.shape[0], axis=0)

            # filtered_result = self._sample_from_clip(
            #     dst_lmdb_env,
            #     pose_each_file,
            #     trans_each_file,
            #     trans_v_each_file,
            #     shape_each_file,
            #     vid_each_file,
            #     disable_filtering,
            #     clean_first_seconds,
            #     clean_final_seconds,
            #     is_test,
            # )
            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file,
                audio_each_file_resample,
                pose_each_file,
                trans_each_file,
                # trans_v_each_file,
                shape_each_file,
                facial_each_file,
                word_each_file,
                vid_each_file,
                f_name,
                emo_each_file,
                sem_each_file,
                disable_filtering,
                clean_first_seconds,
                clean_final_seconds,
                is_test,
            )
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]

        # Print summary
        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(
                colored(
                    "no. of excluded samples: {} ({:.1f}%)".format(
                        n_total_filtered,
                        100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered),
                    ),
                    "cyan",
                )
            )
        dst_lmdb_env.sync()
        dst_lmdb_env.close()

    def idmapping(self, id):
        # map 1,2,3,4,5, 6,7,9,10,11,  12,13,15,16,17,  18,20,21,22,23,  24,25,27,28,30 to 0-24
        if id == 30:
            id = 8
        if id == 28:
            id = 14
        if id == 27:
            id = 19
        return id - 1

    def _sample_from_clip(
        self,
        dst_lmdb_env,
        audio_each_file,
        audio_each_file_resample,
        pose_each_file,
        trans_each_file,
        shape_each_file,
        facial_each_file,
        word_each_file,
        vid_each_file,
        f_name,
        emo_each_file,
        sem_each_file,
        disable_filtering,
        clean_first_seconds,
        clean_final_seconds,
        is_test,
    ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data
        """
        # audio_start = int(self.alignment[0] * self.args.audio_fps)
        # pose_start = int(self.alignment[1] * self.args.pose_fps)
        # logger.info(f"before: {audio_each_file.shape} {pose_each_file.shape}")
        # audio_each_file = audio_each_file[audio_start:]
        # pose_each_file = pose_each_file[pose_start:]
        # trans_each_file =
        # logger.info(f"after alignment: {audio_each_file.shape} {pose_each_file.shape}")
        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps  # assume 1500 frames / 15 fps = 100 s
        if audio_each_file is not None:  #!= []:
            if self.args.audio_rep != "wave16k":
                round_seconds_audio = len(audio_each_file) // self.args.audio_fps  # assume 16,000,00 / 16,000 = 100 s
            elif self.args.audio_rep == "mfcc":
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_fps
            else:
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_sr
            if facial_each_file is not None and facial_each_file != []:
                round_seconds_facial = facial_each_file.shape[0] // self.args.pose_fps
                logger.info(f"audio: {round_seconds_audio}s, pose: {round_seconds_skeleton}s, facial: {round_seconds_facial}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                if round_seconds_skeleton != max_round:
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
            else:
                logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
                # round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
                max_round = max(round_seconds_audio, round_seconds_skeleton)
                if round_seconds_skeleton != max_round:
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")

        clip_s_t, clip_e_t = (
            clean_first_seconds,
            round_seconds_skeleton - clean_final_seconds,
        )  # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = (
            self.args.audio_fps * clip_s_t,
            clip_e_t * self.args.audio_fps,
        )  # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = (
            clip_s_t * self.args.pose_fps,
            clip_e_t * self.args.pose_fps,
        )  # [150,90*15]

        for ratio in self.args.multi_length_training:
            if is_test:  # stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
            else:
                self.args.stride = int(ratio * self.ori_stride)
                cut_length = int(self.ori_length * ratio)

            num_subdivision = 1  # math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")

            if audio_each_file is not None:  #!= []:
                audio_short_length = math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)
                """
                for audio sr = 16000, fps = 15, pose_length = 34, 
                audio short length = 36266.7 -> 36266
                this error is fine.
                """
                logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")

            n_filtered_out = defaultdict(int)
            sample_pose_list = []
            sample_audio_list = []
            sample_audio_resample_list = []
            sample_facial_list = []
            sample_shape_list = []
            sample_word_list = []
            sample_emo_list = []
            sample_sem_list = []
            sample_vid_list = []
            sample_name_list = []
            sample_trans_list = []
            semantic_text_list = []

            for i in range(num_subdivision):  # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length
                sample_pose = pose_each_file[start_idx:fin_idx]

                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]

                # Handle audio
                if self.args.audio_rep is not None:
                    audio_start = clip_s_f_audio + math.floor(i * self.args.stride * self.args.audio_fps / self.args.pose_fps)
                    audio_end = audio_start + audio_short_length
                    sample_audio = audio_each_file[audio_start:audio_end]
                    sample_audio_resample = audio_each_file_resample[audio_start:audio_end]
                else:
                    sample_audio = []
                    sample_audio_resample = audio_each_file_resample

                sample_facial = facial_each_file[start_idx:fin_idx] if self.args.facial_rep is not None and facial_each_file is not None else []
                sample_word = word_each_file[start_idx:fin_idx] if self.args.word_rep is not None and word_each_file is not None else []
                sample_emo = emo_each_file[start_idx:fin_idx] if self.args.emo_rep is not None and emo_each_file is not None else []
                sample_sem = sem_each_file[start_idx:fin_idx] if self.args.sem_rep is not None and sem_each_file is not None else []
                sample_vid = vid_each_file[start_idx:fin_idx] if self.args.id_rep is not None and vid_each_file is not None else []
                sample_name = np.array([f"{f_name}"]) if self.args.id_rep is not None else []  # _{start_idx}_{fin_idx}
                semantic_text = None
                # if is_test:
                #     semantic_labeled_file = os.path.join(self.args.data_path, "semantic_llm", f"{f_name}.txt")
                #     if os.path.exists(semantic_labeled_file):
                #         with open(semantic_labeled_file, "r") as f:
                #             semantic_text = f.read()

                #     text_file = os.path.join(self.args.data_path, "texts", f"{f_name}.txt")
                #     if not os.path.exists(text_file):
                #         with open(text_file, "w") as f:
                #             f.write(self._create_text_from_in_word(sample_word))

                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose).get()
                    is_correct_motion = sample_pose is not None  #!= []
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_audio_list.append(sample_audio)
                        sample_audio_resample_list.append(sample_audio_resample)
                        sample_facial_list.append(sample_facial)
                        sample_shape_list.append(sample_shape)
                        sample_word_list.append(sample_word)
                        sample_vid_list.append(sample_vid)
                        sample_emo_list.append(sample_emo)
                        sample_sem_list.append(sample_sem)
                        sample_name_list.append(sample_name)
                        sample_trans_list.append(sample_trans)
                        semantic_text_list.append(semantic_text)
                    else:
                        n_filtered_out[filtering_message] += 1

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for (
                        pose,
                        audio,
                        audio_resample,
                        facial,
                        shape,
                        word,
                        vid,
                        emo,
                        sem,
                        name,
                        semantic_text,
                        trans,
                    ) in zip(
                        sample_pose_list,
                        sample_audio_list,
                        sample_audio_resample_list,
                        sample_facial_list,
                        sample_shape_list,
                        sample_word_list,
                        sample_vid_list,
                        sample_emo_list,
                        sample_sem_list,
                        sample_name_list,
                        semantic_text_list,
                        sample_trans_list,
                    ):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = [
                            pose,
                            audio,
                            audio_resample,
                            facial,
                            shape,
                            word,
                            emo,
                            sem,
                            vid,
                            name,
                            semantic_text,
                            trans,
                        ]
                        v = pa_serialize(v)  # .to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out


class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        # self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert self.skeletons is not None

        # filtering
        if self.skeletons is not None:  #!= []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons != []:
        #     self.skeletons = self.skeletons.tolist()
        #     for i, frame in enumerate(self.skeletons):
        #         assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False

    def check_pose_diff(self, verbose=False):
        #         diff = np.abs(self.skeletons - self.mean_pose) # 186*1
        #         diff = np.mean(diff)

        #         # th = 0.017
        #         th = 0.02 #0.02  # exclude 3594
        #         if diff < th:
        #             if verbose:
        #             return True
        # #         th = 3.5 #0.02  # exclude 3594
        # #         if 3.5 < diff < 5:
        # #             if verbose:
        # #                 print("skip - check_pose_diff {:.5f}".format(diff))
        # #             return True
        #         else:
        #             if verbose:
        return False

    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
            # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False

import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
from data_loaders.beat2.utils import config
from data_loaders.beat2.utils.cache_utils import calculate_mean_std
from data_loaders.seg.seg_generator import SEG2CacheGenerator
from utils import rotation_conversions as rc
import torch.nn.functional as F
from omegaconf import OmegaConf
import torch as th
from os.path import join as pjoin
import pyarrow as pa
from utils.sampling_utils import sample_to_pose_trans
import lmdb as lmdb
from scipy.interpolate import interp1d
from data_loaders.beat2.data_tools import joints_list
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
from model.rotation2xyz import Rotation2xyz
from data_loaders.beat2.data_tools import joints_indexs_names
import matplotlib.pyplot as plt
import glob

def pa_serialize(obj) -> pa.Buffer:
    """Mimic `pyarrow.serialize(obj).to_buffer()`."""
    return pa.py_buffer(pickle.dumps(obj, protocol=5))  # protocol 5 = Py3.8+


def pa_deserialize(buf: pa.Buffer):
    """Mimic `pyarrow.deserialize(buf)`."""
    # `memoryview` keeps it zero-copy if `buf` already points to shared memory
    return pickle.loads(memoryview(buf))


def extract_word_timestamps(in_word, lang_model):
    def remove_consecutive_duplicates(arr):
        arr = np.array(arr)
        new_arr = [arr[0]]
        indices = [(0, 0)]  # Stores (start, end) indices for each unique token

        start = 0
        for i in range(1, len(arr)):
            if arr[i] != arr[i - 1]:
                indices[-1] = (start, i - 1)
                new_arr.append(arr[i])
                start = i
                indices.append((i, i))
            else:
                indices[-1] = (indices[-1][0], i)

        return new_arr, indices

    # Flatten the tensor if needed
    if len(in_word.shape) > 1:
        in_word = in_word.squeeze().tolist()

    new_in_word, indices = remove_consecutive_duplicates(in_word)
    words = [lang_model.index2word[t] for t in new_in_word]

    special_tokens = {"<PAD>", "<SOS>", "<EOS>", ""}
    word_timestamps = []
    for word, (start, end) in zip(words, indices):
        if word in special_tokens:
            continue
        word_timestamps.append({"word": word, "start": start, "end": end})

    return word_timestamps


def find_seg_code_info(in_word, lang_model, seg_dataset, llm_label_res, nadav=None, only_nadav=False, word_timestamps=None):

    if word_timestamps is not None:
        sentences_time_info = word_timestamps
    else:
        sentences_time_info = extract_word_timestamps(in_word, lang_model)
    num_words_in_input = len(sentences_time_info)

    insert_sg_info = []
    i = 0
    modified_s_0 = re.sub(r"\s([,.!?])", r"\1", llm_label_res[i])
    modified_s_1 = re.sub(r"(?<!\s)\(", " (", modified_s_0)
    print(modified_s_1)

    pattern = re.compile(r"\((.*?)\)")
    matches = pattern.finditer(modified_s_1)

    words_num_in_brackets = 0
    for match in matches:
        print(match)
        content = match.group(1)
        start_position = match.start()
        substring_before_bracket = modified_s_1[:start_position]
        words_before_bracket = substring_before_bracket.split()
        last_word = words_before_bracket[-1]
        word_index = len(words_before_bracket) - 1 - words_num_in_brackets

        assert word_index < len(sentences_time_info), (
            f"word_index {word_index} out of range (have {len(sentences_time_info)} words)"
        )
        expected_word = sentences_time_info[word_index]["word"]
        assert expected_word == last_word, (
            f"Word mismatch at index {word_index}: semantic text has '{last_word}' but tokenized has '{expected_word}'"
        )
        if word_index >= len(sentences_time_info) or expected_word != last_word:
            print(
                f"Warning: word alignment mismatch at word_index {word_index} "
                f"(expected '{last_word}'), skipping remaining gesture tags for this sentence"
            )
            break
        words_num_in_brackets += len(content.split())
        if word_index >= num_words_in_input:
            continue
        sg_index = int(content.split()[0])
        if sg_index not in seg_dataset.data_info.keys():
            print("sg_index {} not in gestures".format(sg_index))
            # words_num_in_brackets += len(content.split())
            continue

        sg_info = seg_dataset.gestures[sg_index]
        print(i, word_index, last_word)
        print(len(sentences_time_info))
        # choose one option
        sg_motion_num = len(sg_info)
        if sg_motion_num == 0:
            print(f"didt translate motion num: {sg_motion_num}")
            continue
        if nadav is not None:
            name_motion = sg_info[0]["file_name"].split("-")[0]
            if name_motion not in nadav:
                if only_nadav:
                    print(f"name_motion: {name_motion} not in nadav folder and only_nadav is True, skipping")
                    continue
                choice_index = np.random.randint(0, sg_motion_num)
                print(f"name_motion: {name_motion} not in nadav folder choosing random")
                # continue
            else:
                choice_index = nadav[name_motion][0]
                print(f"name_motion: {name_motion}, choice_index: {choice_index}")
                del nadav[name_motion][0]
        else:
            choice_index = np.random.randint(0, sg_motion_num)

        # Get joints in motion for this gesture (with caching)
        name = sg_info[choice_index]["file_name"][:-4]
        joints_rot6d_mask = seg_dataset.get_joints_in_motion_for_gesture(sg_index)

        insert_motion_dict = {
            "semantic_gesture_index": sg_index,
            "semantic_gesture_label": content[content.index(" ") + 1 :],
            "semantic_gesture_info": sg_info,
            "choice_index": choice_index,
            "sentence_index": i,
            "word_index": word_index,
            "last_word": last_word,
            "start_code": sentences_time_info[word_index]["start"],
            "end_code": sentences_time_info[word_index]["end"],
            "joints_rot6d_mask": joints_rot6d_mask,
        }
        insert_sg_info.append(insert_motion_dict)

    return insert_sg_info


class SegDataset(Dataset):
    """
    A dataset that reads an XLSX file containing columns:
    [Label, Description, Contextual Meaning, Example],
    and associates each row with a corresponding motion .npz file
    in a specified folder. It caches the mapping in a pickle file
    so that repeated parsing is avoided.
    """

    # Class-level cache for gestures to avoid reloading on each instance
    _gestures_cache = None
    _gestures_cache_key = None

    @classmethod
    def clear_gestures_cache(cls, cache_path=None):
        """Clear the gestures cache to force reloading on next initialization"""
        cls._gestures_cache = None
        cls._gestures_cache_key = None
        if cache_path and os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"[SegDataset] Gestures cache file removed: {cache_path}")
        print("[SegDataset] Gestures cache cleared")

    def __init__(
        self,
        split: str = "train",
        build_cache: bool = True,
        device=0,
        xlsx_path: str = "./datasets/SeG_SMPLX/SeG_list.xlsx",
        npz_folder: str = "./datasets/SeG_SMPLX/seg_dataset_new_skeleton",
        cache_path: str = "./datasets/SeG_SMPLX/seg_dataset_cache.pkl",
        transform=None,
        num_timesteps=100,
        config_seg_opt_path=None,
        pose_norm=True,
        use_trans=True,
        mean=None,
        std=None,
        diffusion=None,
        model=None,
    ):
        """
        Args:
            xlsx_path   : Path to the XLSX file containing the table.
            npz_folder  : Folder containing the .npz motion files.
            cache_path  : Where to store/load the cache of the dataset mapping.
            transform   : Optional transform or processing to apply on loaded data.
        """
        args = config.parse_args()
        self.args = args
        self.device = device
        self.xlsx_path = xlsx_path
        self.npz_folder = npz_folder
        self.cache_path = cache_path
        self.transform = transform
        self.num_timesteps = num_timesteps
        self.rank = 0
        self.joints = 55
        self.vae_test_len = 32

        # If cache exists, load it. Otherwise parse XLSX and create it.
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.data_info = pickle.load(f)
            print(f"[SegDataset] Loaded cache from {self.cache_path}")
        else:
            print("[SegDataset] No cache found. Building dataset mapping from XLSX...")
            self.data_info = self._build_dataset_mapping()
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.data_info, f)
            print(f"[SegDataset] Cache saved to {self.cache_path}")

        seg_cache_folder_path = os.path.join(self.args.root_path, os.path.dirname(self.args.cache_path) , "seg")
        cache_folder_path = os.path.join(seg_cache_folder_path, split, f"{args.pose_rep}_cache")
        # Build cache

        self.cache_generator = SEG2CacheGenerator(args, device)
        if build_cache:
            self.smplx = self.cache_generator.get_smplx()
            self.cache_generator.build_cache(cache_folder_path, data_folder=args.data_seg_path, split=split, force_build=self.args.new_cache)
            # Calculate mean and the std values of dataset
            calculate_mean_std(seg_cache_folder_path)

        # Load cache
        self.lmdb_env = lmdb.open(cache_folder_path, readonly=True, lock=False)
        self.name_to_key = {}
        self.name_to_key_sg = {}
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]
        for idx in range(self.n_samples):
            with self.lmdb_env.begin(write=False) as txn:
                key = "{:005}".format(idx).encode("ascii")
                sample = txn.get(key)
                sample = pa_deserialize(sample)
                tar_pose, in_audio, in_audio_resample, in_facial, in_shape, in_word, emo, sem, vid, name, text_semantic, trans = sample
                name = str(name[0]).split(".")[0]
                self.name_to_key[name] = idx
        if mean is None:
            # self.base_seg_folder = os.path.dirname(self.npz_folder)
            # if not os.path.exists(self.mean_seg_path) or not os.path.exists(self.std_seg_path):
            #     # calculate_mean_std(args.cache_path + "_seg")

            #     mean = np.load(os.path.join(args.root_path, args.cache_path + "_seg", "Mean.npy"))
            #     std = np.load(os.path.join(args.root_path, args.cache_path + "_seg", "Std.npy"))
            cache_root_folder_path: str = os.path.join(args.root_path, args.cache_path)
            mean = np.load(os.path.join(cache_root_folder_path, "Mean.npy"))
            std = np.load(os.path.join(cache_root_folder_path, "Std.npy"))

        self.mean = mean
        self.std = std
        self.std[self.std == 0] = 1

        self.seg_fps = 250 / 3
        self.get_gestures()
        mean_vel_path = seg_cache_folder_path + f"/mean_vel_{args.pose_rep}.npy"
        if os.path.exists(mean_vel_path):
            self.mean_vel = np.load(mean_vel_path)
        else:
            self._calculate_mean_velocity(mean_vel_path)

        self.pose_norm = pose_norm
        self.use_trans = use_trans

        for sg_index in self.gestures:
            if len(self.gestures[sg_index]) > 0:
                name = self.gestures[sg_index][0]["file_name"][:-4].split("-")[0]
                self.name_to_key_sg[name] = sg_index

        # Initialize joints in motion cache
        self.cache_dir_joints_in_motion = "./datasets/beat_cache/seg/cache_joints_in_motion"
        os.makedirs(self.cache_dir_joints_in_motion, exist_ok=True)

        # Load existing cache if available
        self._load_joints_in_motion_cache()
        if len(self.joints_in_motion_cache) == 0:
            self._precompute_joints_in_motion_for_all_gestures()

        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][1]] = 1
        self.integration_case = 1

    def _load_joints_in_motion_cache(self):
        """Load joints in motion cache from disk"""
        cache_file = os.path.join(self.cache_dir_joints_in_motion, "joints_in_motion_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.joints_in_motion_cache = pickle.load(f)
                print(f"[SegDataset] Loaded joints in motion cache with {len(self.joints_in_motion_cache)} entries")
            except Exception as e:
                print(f"[SegDataset] Failed to load joints in motion cache: {e}")
                self.joints_in_motion_cache = {}
        else:
            self.joints_in_motion_cache = {}

    def _save_joints_in_motion_cache(self):
        """Save joints in motion cache to disk"""
        cache_file = os.path.join(self.cache_dir_joints_in_motion, "joints_in_motion_cache.pkl")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.joints_in_motion_cache, f)
            print(f"[SegDataset] Saved joints in motion cache with {len(self.joints_in_motion_cache)} entries")
        except Exception as e:
            print(f"[SegDataset] Failed to save joints in motion cache: {e}")

    def plot_chosen_joints_on_skeleton(
        self,
        joints_take_part_in_motion_indexs,
        motion_data_xyz=None,
        frame_idx=0,
        save_path=None,
        name_motion=None,
        figsize=(6, 8),
        marker_size=80,
        bone_lw=2.0,
        projection_axes=("x", "y"),
    ):
        """
        Plot the chosen joints in red on the SMPLX skeleton.

        Args:
            joints_take_part_in_motion_indexs: Array of joint indices that take part in motion
            motion_data_xyz: Optional motion data in XYZ format (T, J, 3)
            frame_idx: Frame index to visualize (default: 0)
            save_path: Optional path to save the plot
            figsize: Figure size tuple
            marker_size: Size of joint markers
            bone_lw: Line width for bones
            projection_axes: Which axes to project onto (e.g., ("x", "z"))
        """
        # Try to load SMPLX skeleton structure, fallback to default if not available
        parents = None
        J = 55

        # Try multiple possible paths for SMPLX model
        possible_paths = [
            "./datasets/hub/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz",
            "./smplx_models/smplx/SMPLX_NEUTRAL_2020.npz",
        ]

        for smpl_fname in possible_paths:
            try:
                if os.path.exists(smpl_fname):
                    smpl_data = np.load(smpl_fname, encoding="latin1")
                    parents = smpl_data["kintree_table"][0].astype(np.int32)
                    print(f"Loaded SMPLX skeleton from: {smpl_fname}")
                    break
            except Exception as e:
                print(f"Failed to load {smpl_fname}: {e}")
                continue

        # Fallback to default skeleton structure if SMPLX file not found
        if parents is None:
            print("Warning: SMPLX model not found, using default skeleton structure")
            parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15] + [-1] * 30)
            parents = parents[:J]  # Ensure we have exactly J parents

        # Handle motion data
        if motion_data_xyz is None:
            print("Warning: No motion data provided, using default T-pose positions")
            joint_positions = self._get_default_tpose_positions()
            if joint_positions.shape[0] < J:
                # Extend with zeros if needed
                padding = np.zeros((J - joint_positions.shape[0], 3))
                joint_positions = np.vstack([joint_positions, padding])
        else:
            # Ensure frame_idx is within bounds
            if frame_idx >= motion_data_xyz.shape[0]:
                frame_idx = 0
                print(f"Warning: frame_idx {frame_idx} out of bounds, using frame 0")
            joint_positions = motion_data_xyz[frame_idx]  # (J, 3)

        # --- Project to 2D ---
        axis_map = {"x": 0, "y": 1, "z": 2}
        if projection_axes[0] not in axis_map or projection_axes[1] not in axis_map:
            raise ValueError("projection_axes must contain 'x','y' or 'z'")
        ax0 = axis_map[projection_axes[0]]
        ax1 = axis_map[projection_axes[1]]

        pts2d = joint_positions[:, [ax0, ax1]]  # (J, 2)

        # Center and scale for nicer figure
        min_xy = pts2d.min(axis=0)
        max_xy = pts2d.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        size = (max_xy - min_xy).max()
        if size == 0:
            size = 1.0
        pts2d_centered = (pts2d - center) / size  # normalized to roughly [-0.5,0.5]
        display_scale = 1.0
        pts_display = pts2d_centered * display_scale

        # --- Create plot ---
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Draw bones using parents, only where parent index is valid
        for child_idx in range(min(len(parents), J)):
            p = int(parents[child_idx])
            if p < 0 or p >= J:
                continue
            p0 = pts_display[p]
            p1 = pts_display[child_idx]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=bone_lw, solid_capstyle="round", zorder=1, color="black", alpha=0.7)

        # Draw all joints faintly
        ax.scatter(pts_display[:, 0], pts_display[:, 1], s=20, alpha=0.6, zorder=2, color="blue")

        # Highlight chosen joints in red
        chosen_idx = np.array(joints_take_part_in_motion_indexs, dtype=int)
        valid_mask = (chosen_idx >= 0) & (chosen_idx < J)
        if not np.all(valid_mask):
            print(f"Warning: ignoring out-of-range chosen joint indices: {chosen_idx[~valid_mask]}")
            chosen_idx = chosen_idx[valid_mask]

        if chosen_idx.size > 0:
            coords = pts_display[chosen_idx]
            ax.scatter(coords[:, 0], coords[:, 1], s=marker_size, c="red", zorder=3, alpha=0.9)

            # # Add joint labels (index and name if available)
            # for i, idx in enumerate(chosen_idx):
            #     if idx in joints_indexs_names:
            #         label = f"{idx}: {joints_indexs_names[idx]}"
            #     else:
            #         label = str(idx)
            #     ax.text(coords[i, 0], coords[i, 1], label, fontsize=8, color="red", ha="center", va="center", weight="bold")

        # Set tight bounding box with padding
        pad = 0.6
        # ax.set_xlim(-0.5 * pad, 0.5 * pad)
        # ax.set_ylim(-0.5 * pad, 0.5 * pad)

        # Add title
        title = f"Chosen Joints in {name_motion.replace('gesture_', ' ')}" if name_motion else "Chosen Joints in Motion"
        ax.set_title(title, fontsize=12, pad=20)

        # Save plot if path provided
        if save_path:
            # Print chosen joint names
            # for joint_idx in joints_take_part_in_motion_indexs:
            #     if joint_idx in joints_indexs_names:
            #     else:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
            print(f"Saved 2D skeleton image to {save_path}")

        # Show the plot
        plt.show()

        return fig, ax

    # def recalculate_joints_take_part_in_motion_mask(self, motion_data_xyz, name_motion=None, frame_idx=0):
    #     mask = self.find_joints_take_part_in_motion_mask(motion_data_xyz, name_motion=name_motion, frame_idx=frame_idx)
    #     return mask

    def find_joints_take_part_in_motion_mask(self, motion_data_xyz, use_parent=True, name_motion=None, frame_idx=0):
        T, J, _ = motion_data_xyz.shape

        smpl_fname = "./datasets/hub/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"  # todo - remove hard-coded
        smpl_data = np.load(smpl_fname, encoding="latin1")
        parents = smpl_data["kintree_table"][0].astype(np.int32)

        # Calculate spatial entropy for each joint based on position variability
        joint_mean_veloceties = []
        joint_entropies = []
        for joint_idx in range(J):
            # Get joint positions over time: [T, 3]
            joint_positions = motion_data_xyz[:, joint_idx, :]  # .detach().cpu().numpy()  # [T, 3]

            # Calculate spatial variability using standard deviation of positions
            # pos_std = np.std(joint_positions, axis=0)  # [3] - std for x, y, z
            D = torch.cdist(joint_positions, joint_positions).cpu().numpy()
            spatial_variability = np.max(D)  # Max std across x, y, z
            joint_entropies.append(spatial_variability)
            # vel = np.linalg.norm(joint_positions[1:] - joint_positions[:-1], axis=1) / (1 / 30)
            # joint_mean_veloceties.append(np.sqrt(np.mean(vel[np.argsort(vel)][-30:] ** 2)) / self.mean_vel[joint_idx])
            # joint_mean_veloceties.append(np.max(vel) / self.mean_vel[joint_idx])
            #  [(joints_indexs_names[i], v, threshold[i]) for i, v in enumerate(joint_entropies)]

        # joint_mean_veloceties = np.array(joint_mean_veloceties)
        joint_entropies = np.array(joint_entropies)
        # Use threshold to select joints with high spatial entropy
        # threshold = 1.7
        threshold = [0.1] * 16 + [0.1] * 4 + [0.3] * 2 + [0.1] * 3 + [0.3] * 30

        # Create mask: joints with high spatial entropy are considered "in motion"
        mask = torch.zeros((J), dtype=torch.bool, device=motion_data_xyz.device)
        # mask[joint_mean_veloceties > threshold] = True
        for joint_idx in range(J):
            if joint_entropies[joint_idx] > threshold[joint_idx]:
                mask[joint_idx] = True  # Mark as active for all timesteps
                if use_parent:
                    mask[parents[joint_idx]] = True  # Mark as active for all timesteps

        joints_take_part_in_motion_indexs = np.where(mask.detach().cpu().numpy())[0]
        if name_motion is not None:
            print(f"Found {len(joints_take_part_in_motion_indexs)} joints taking part in motion {name_motion}:")
            print([(joints_indexs_names[i], v, threshold[i]) for i, v in enumerate(joint_entropies)])
            # Note: joints_indexs_names would need to be defined or passed as parameter
            for j in joints_take_part_in_motion_indexs:
                if j in joints_indexs_names:
                    print(f"  {j}: {joints_indexs_names[j]}")

                    # Plot chosen joints on skeleton (using default positions since we don't have dataset)

            plot_save_path = os.path.join(self.cache_dir_joints_in_motion, f"{name_motion}_chosen_joints_visualization.png")

            self.plot_chosen_joints_on_skeleton(
                joints_take_part_in_motion_indexs,
                motion_data_xyz=motion_data_xyz.detach().cpu().numpy() if motion_data_xyz is not None else None,
                # frame_idx=0,
                save_path=plot_save_path,
                projection_axes=("x", "y"),
                name_motion=name_motion,
                frame_idx=frame_idx,
            )
            # SAVE [(joints_indexs_names[i], v, threshold[i]) for i, v in enumerate(joint_entropies)] IN TEXT FILE
            with open(os.path.join(self.cache_dir_joints_in_motion, f"{name_motion}_joints_entropies.txt"), "w") as f:
                for i, v in enumerate(joint_entropies):
                    f.write(f"{joints_indexs_names[i]}: {v}, threshold: {threshold[i]}\n")

        return mask

    def calc_joints_take_part_in_motion(self, motion_data, use_parent=True, min_frames=5, return_rot6d_mask=True, cache_key=None, frame_idx=0):
        """Calculate joints that take part in motion based on spatial entropy (position variability)"""

        # Check if we have cached result
        if cache_key is not None and cache_key in self.joints_in_motion_cache:
            print(f"[SegDataset] Using cached joints in motion for {cache_key or 'motion'}")
            cached_result = self.joints_in_motion_cache[cache_key]
            return cached_result["mask"], cached_result["motion_data_xyz"]

        # Convert motion data to xyz coordinates
        # Note: This requires access to rot2xyz method, which might need to be passed as parameter
        # For now, we'll assume motion_data is already in the right format
        if hasattr(self, "device"):
            device = self.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        motion_data_xyz = Rotation2xyz(device=self.device)(
            torch.Tensor(motion_data).to(torch.float32).to(self.device), "rot_6d", data=self, use_global=False
        )[0]
        mask = self.find_joints_take_part_in_motion_mask(motion_data_xyz, use_parent=use_parent, name_motion=cache_key, frame_idx=frame_idx)

        if return_rot6d_mask:
            mask = torch.repeat_interleave(mask, 6)

        # Cache the result
        self.joints_in_motion_cache[cache_key] = {
            "mask": mask,
            "motion_data_xyz": motion_data_xyz,
            # "joints_take_part_in_motion_indexs": joints_take_part_in_motion_indexs,
        }

        return mask, motion_data_xyz

    def get_joints_in_motion_for_gesture(self, sg_index, motion_data=None):
        """Get joints in motion for a specific gesture, with caching"""
        # cache_key = f"gesture_{sg_index}_{choice_index}"
        sg_info = self.gestures[sg_index]
        name = sg_info[0]["file_name"][:-4]
        cache_key = f"gesture_{sg_index}_{name}"
        if cache_key in self.joints_in_motion_cache:
            return self.joints_in_motion_cache[cache_key]["mask"]

        return None

    def get_joints_take_part_in_motion_from_cache(self, motion_name: str) -> list:
        """
        Get joint indices that take part in motion from correlation cache.

        Args:
            motion_name: Name of the motion (e.g., "HANDS_BEAST-1")

        Returns:
            List of joint indices that take part in the motion
        """
        try:
            # Search for cache key that contains the motion name
            # Cache keys are in format: "gesture_{sg_index}_{name}"
            matching_key = None
            for cache_key in self.joints_in_motion_cache.keys():
                if motion_name in cache_key:
                    matching_key = cache_key
                    break

            if matching_key is None:
                print(f"⚠️ No cache key found containing motion name: {motion_name}")
                print(f"   Available cache keys: {list(self.joints_in_motion_cache.keys())[:5]}...")  # Show first 5 keys
                return list(range(25, 55))  # Fallback to hand joints

            print(f"📁 Found matching cache key: {matching_key}")
            cache_data = self.joints_in_motion_cache[matching_key]

            # Extract joint indices from cache data
            if "joints_take_part_in_motion_indexs" in cache_data:
                joint_indices = cache_data["joints_take_part_in_motion_indexs"]
                print(f"✅ Found {len(joint_indices)} joints for motion {motion_name}: {joint_indices}")
                return joint_indices
            else:
                print(f"⚠️ No joint indices found in cache for motion: {motion_name}")
                return list(range(25, 55))  # Fallback to hand joints

        except Exception as e:
            print(f"❌ Error loading joint indices from cache: {e}")
            return list(range(25, 55))  # Fallback to hand joints

    def _precompute_joints_in_motion_for_all_gestures(self):
        """Precompute joints in motion for all gestures to speed up runtime"""
        print("[SegDataset] Precomputing joints in motion for all gestures...")
        processed = 0

        for sg_index in self.gestures:
            for choice_index, gesture_data in enumerate(self.gestures[sg_index]):
                name = gesture_data["file_name"][:-4]
                cache_key = f"gesture_{sg_index}_{name}"

                # Skip if already cached
                if cache_key in self.joints_in_motion_cache:
                    # processed += 1
                    continue

                # Get motion data from gesture
                if "poses_6d" in gesture_data:
                    motion_data = gesture_data["poses_6d"].permute(0, 2, 1).unsqueeze(2)

                    # Calculate joints in motion
                    mask, _ = self.calc_joints_take_part_in_motion(motion_data, cache_key=cache_key)

                    # # Cache the result
                    # self.joints_in_motion_cache[cache_key] = {"joints_rot6d_mask": mask}

        # Save the complete cache
        self._save_joints_in_motion_cache()
        print(f"[SegDataset] Completed precomputing joints in motion for {processed} gestures")

    def _build_dataset_mapping(self):
        """
        Reads the XLSX file and constructs a list of dictionaries.
        Each entry will have the relevant columns plus the path to
        the corresponding .npz file.

        We assume:
          - The XLSX has columns: 'Label', 'Description', 'Contextual Meaning', 'Example'.
          - The .npz files are named in some pattern, e.g. 'ARM_RAISE_HIGH_LEVEL-[SOME INDEX].npz'
          - If label is 'ARM FLEX [1]', the file is 'ARM_RAISE_HIGH_LEVEL-[SOME INDEX].npz'
            (You can adapt the logic below to match your naming convention.)
        """
        df = pd.read_excel(self.xlsx_path, header=2)

        data_dict = {}
        for idx, row in df.iterrows():
            label = row["Label"]  # e.g. "ARM FLEX [1]"
            sem = row["Semantics-Aware Index"]
            description = row["Description"]
            context = row["Contextual Meaning"]

            if "[1]" in label:
                label = label[:-4]

            name_file = f"{label.split(" ")[0]}"
            for i in label.split(" ")[1:]:
                name_file += "_"
                name_file += i

            npz_names = [file_name for file_name in os.listdir(self.npz_folder) if name_file.replace("'", "").replace("-", "_") in file_name]

            entry = {
                "Label": label,
                "Description": description,
                "ContextualMeaning": context,
                "npz_names": npz_names,
            }
            data_dict[sem] = entry

        return data_dict

    def __len__(self):
        return self.n_samples

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def get_gestures(self):
        # Create cache file path based on npz_folder and seg_fps
        cache_dir = os.path.dirname(self.npz_folder)
        cache_filename = f"gestures_cache.pkl"
        cache_path = os.path.join(cache_dir, cache_filename)

        # Check if cache file exists and is valid
        cache_valid = False
        if os.path.exists(cache_path):
            try:
                # Check if cache is newer than the data_info cache
                cache_mtime = os.path.getmtime(cache_path)
                data_info_mtime = os.path.getmtime(self.cache_path) if os.path.exists(self.cache_path) else 0

                if cache_mtime > data_info_mtime:
                    # Load from cache file
                    with open(cache_path, "rb") as f:
                        cached_data = pickle.load(f)
                        self.gestures = cached_data["gestures"]
                        print(f"[SegDataset] Loaded gestures from cache file: {cache_path}")
                        print(f"[SegDataset] Cached gestures with {len(self.gestures)} gesture types")
                        cache_valid = True
            except Exception as e:
                print(f"[SegDataset] Failed to load gestures cache: {e}")

        if not cache_valid:
            print(f"[SegDataset] Loading gestures from NPZ files...")
            self.gestures = {}
            for sg_index in self.data_info:
                self.gestures[sg_index] = []
                for npz_name in self.data_info[sg_index]["npz_names"]:
                    data = {"file_name": npz_name}
                    npz_path = os.path.join(self.npz_folder, npz_name)
                    dict_data = np.load(npz_path, allow_pickle=True)
                    if self.seg_fps == 60:
                        tar_pose_raw = torch.tensor(dict_data["poses"])[::2]
                        tar_trans = torch.tensor(dict_data["trans"])[::2].to(self.rank)
                        # tar_exps = torch.tensor(dict_data["expressions"])[::2].to(self.rank)
                    else:
                        tar_pose_raw = torch.tensor(dict_data["poses"])
                        tar_trans = torch.tensor(dict_data["trans"]).to(self.rank)
                        # tar_exps = torch.tensor(dict_data["expressions"]).to(self.rank)
                    tar_pose = tar_pose_raw[:, :165].to(self.rank)
                    bs, n, j = 1, tar_pose.shape[0], self.joints

                    data["poses"] = tar_pose[:n]
                    tar_pose_matrix = rc.axis_angle_to_matrix(tar_pose.reshape(-1, j, 3))
                    tar_poses_6d = rc.matrix_to_rotation_6d(tar_pose_matrix).reshape(bs, -1, j * 6)

                    tar_poses_all = (
                        torch.cat([tar_poses_6d, torch.zeros((1, n, 3)).to(tar_trans.device), torch.zeros((1, n, 4)).to(tar_trans.device)], dim=-1)
                        .cpu()
                        .numpy()
                    )

                    # SIGMA = 2          # frames; increase for heavier blur
                    # poses_smoothed = gaussian_filter1d(tar_poses_all, sigma=SIGMA, axis=0, mode="nearest")
                    # from scipy.signal import savgol_filter

                    # WINDOW = 9         # odd integer ≥ 5 and < T
                    # POLY   = 3         # polynomial degree (≤ WINDOW-1)
                    # tar_poses_all[0] = savgol_filter(tar_poses_all[0], window_length=WINDOW, polyorder=POLY, axis=0, mode="interp")

                    # FPS = 90  # frames per second of your data
                    CUTOFF_HZ = 5  # everything above CUTOFF_HZ is considered "noise"
                    b, a = butter(N=4, Wn=CUTOFF_HZ / (0.5 * self.seg_fps), btype="low")

                    tar_poses_all[0] = filtfilt(b, a, tar_poses_all[0], axis=0, padlen=15)

                    length_pose = 90

                    _, T, D = tar_poses_all.shape
                    old_idx = np.arange(T)
                    new_idx = np.linspace(0, T - 1, length_pose)
                    f = interp1d(old_idx, tar_poses_all, axis=1, kind="linear")
                    pose_interpolate = f(new_idx)

                    poses_6d_no_normalized = torch.tensor(pose_interpolate[:, :length_pose]).detach().cpu().numpy()
                    pose_interpolate_normalized = poses_6d_no_normalized.copy()
                    pose_interpolate_normalized[:, :, :330] = (poses_6d_no_normalized[:, :, :330] - self.mean[:330]) / self.std[:330]
                    # tar_poses_all[:, :, :330] = (tar_poses_all[:, :, :330] - self.mean[:330]) / self.std[:330]

                    # tar_poses_all[:, :, :333] = (tar_poses_all[:, :, :333] - self.mean[:333]) / self.std[:333]
                    # tar_poses_all = (tar_poses_all - self.mean) / self.std
                    # data["poses_6d_no_normalized"] = torch.tensor(poses_6d_no_normalized[:, :length_pose])
                    data["poses_6d"] = torch.tensor(pose_interpolate_normalized[:, :length_pose])
                    motion_data_xyz = Rotation2xyz(device=self.device)(
                        torch.Tensor(data["poses_6d"].permute(0, 2, 1).unsqueeze(2)).to(torch.float32).to(self.device),
                        "rot_6d",
                        data=self,
                        use_global=False,
                    )[0]
                    data["poses_6d_xyz"] = motion_data_xyz
                    # data["poses_6d"] = torch.tensor(pose_interpolate[:, :n])
                    data["length"] = length_pose  # tar_pose.shape[1]

                    self.gestures[sg_index].append(data)

            # Save to cache file
            try:
                cache_data = {
                    "gestures": self.gestures,
                    "cache_info": {
                        "npz_folder": self.npz_folder,
                        "seg_fps": self.seg_fps,
                        "data_info_keys": sorted(self.data_info.keys()),
                        "created_at": pd.Timestamp.now().isoformat(),
                    },
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"[SegDataset] Saved gestures cache to: {cache_path}")
                print(f"[SegDataset] Cached gestures with {len(self.gestures)} gesture types")
            except Exception as e:
                print(f"[SegDataset] Failed to save gestures cache: {e}")

    def _calculate_mean_velocity(self, save_path):
        smplx_model = self.smplx

        dir_p = "./datasets/SeG_SMPLX/seg_dataset_new_skeleton/"
        all_list = []
        from tqdm import tqdm

        for tar in tqdm(os.listdir(dir_p)):
            if tar.endswith(".npz"):
                m_data = np.load(dir_p + tar, allow_pickle=True)
                betas, poses, trans, exps = (
                    m_data["betas"],
                    m_data["poses"],
                    m_data["trans"],
                    m_data["expressions"],
                )
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).to(self.device).float()
                poses = torch.from_numpy(poses.reshape(n, c)).to(self.device).float()
                exps = torch.from_numpy(exps.reshape(n, 100)).to(self.device).float()
                trans = torch.from_numpy(trans.reshape(n, 3)).to(self.device).float()
                max_length = self.args.pose_length
                s, r = n // max_length, n % max_length
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = smplx_model(
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
                        joints = smplx_model(
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
                dt = 1 / self.seg_fps
                # first steps is forward diff (t+1 - t) / dt
                init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
                # middle steps are second order (t+1 - t-1) / 2dt
                middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
                # last step is backward diff (t - t-1) / dt
                final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
                vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1).permute(1, 0).reshape(n, 55, 3)
                # .permute(1, 0).reshape(n, 55, 3)
                vel_seq_np = vel_seq.cpu().numpy()
                vel_joints_np = np.linalg.norm(vel_seq_np, axis=2)  # n * 55
                all_list.append(vel_joints_np)
        self.mean_vel = np.mean(np.concatenate(all_list, axis=0), axis=0)  # 55

        np.save(save_path, self.mean_vel)

    # def __getitem__(self, idx):
    #     """
    #     Loads the .npz file for the given index, plus the label metadata.
    #     Applies any optional transform.
    #     """
    #     item_info = self.data_dict[idx]
    #     npz_names = item_info["npz_names"]

    #     if not os.path.exists(npz_names):
    #         raise FileNotFoundError(f"Motion file not found: {npz_names}")

    #     sg_motion_num = len(npz_names)
    #     choice_index = np.random.randint(0, sg_motion_num)

    #     npz_path = os.path.join(self.npz_folder, npz_names[choice_index])
    #     data = np.load(npz_path)
    #     # Convert to torch tensors if needed
    #     motion_tensor = torch.tensor(data["motion"], dtype=torch.float32)  # Example usage

    #     sample = {
    #         "motion": motion_tensor,
    #         "label": item_info["Label"],
    #         "description": item_info["Description"],
    #         "contextual_meaning": item_info["ContextualMeaning"],
    #         "example": item_info["Example"],
    #     }

    #     if self.transform:
    #         sample = self.transform(sample)

    #     return sample

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
            # item_info = self.data_info[idx]
            # npz_names = item_info["npz_names"]

            # if not os.path.exists(npz_names):
            #     raise FileNotFoundError(f"Motion file not found: {npz_names}")

            # sg_motion_num = len(npz_names)
            # choice_index = np.random.randint(0, sg_motion_num)

            # npz_path = os.path.join(self.npz_folder, npz_names[choice_index])
            # data = np.load(npz_path)
            # # Convert to torch tensors if needed
            # motion_tensor = torch.tensor(data["motion"], dtype=torch.float32)  # Example usage
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

    def getitem_by_name(self, name):
        with self.lmdb_env.begin(write=False) as txn:
            idx = self.name_to_key[name]
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pa_deserialize(sample)
            tar_pose_orig, in_audio, in_audio_resample, in_facial, in_shape, in_word, emo, sem, vid, name, text_semantic, trans = sample
            length_pose = 90
            FPS = 30  # frames per second of your data
            CUTOFF_HZ = 5  # everything above CUTOFF_HZ is considered “noise”
            b, a = butter(N=4, Wn=CUTOFF_HZ / (0.5 * FPS), btype="low")

            tar_pose_orig_smooth = filtfilt(b, a, tar_pose_orig, axis=0, padlen=15)

            tar_pose = np.zeros((196, 165))
            tar_pose[:length_pose, :] = tar_pose_orig_smooth
            emo = torch.from_numpy(np.array([-1])).int()
            sem = torch.from_numpy(np.array([-1])).float()
            in_audio = torch.from_numpy(np.array([-1])).float()
            in_audio = torch.zeros([104533, 2])
            in_audio_resample = torch.from_numpy(np.array([-1])).float()
            in_word = torch.zeros([196]).int()  # torch.from_numpy(np.array([-1])).float()
            tar_pose = torch.from_numpy(tar_pose).float()
            trans = torch.zeros([196, 3])  # torch.from_numpy(trans).float()
            in_facial = torch.from_numpy(np.array([-1])).float()
            vid = torch.from_numpy(vid).float()
            in_shape = torch.from_numpy(in_shape).float()
            if text_semantic is None:
                text_semantic = ""
            in_text = ""
            # item_info = self.data_info[idx]
            # npz_names = item_info["npz_names"]

            # if not os.path.exists(npz_names):
            #     raise FileNotFoundError(f"Motion file not found: {npz_names}")

            # sg_motion_num = len(npz_names)
            # choice_index = np.random.randint(0, sg_motion_num)

            # npz_path = os.path.join(self.npz_folder, npz_names[choice_index])
            # data = np.load(npz_path)
            # # Convert to torch tensors if needed
            # motion_tensor = torch.tensor(data["motion"], dtype=torch.float32)  # Example usage
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
        tar_contact = torch.zeros([196, 4])  # tar_pose_raw[:, 165:169]
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
            latent_all[:, :330] = (latent_all[:, :330] - self.mean[:330]) / self.std[:330]

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

    # def calculate_mean_std(self) -> None:
    #     """This function calculate the mean and std of dataset."""

    #     data_list = []
    #     data_list_trans = []
    #     for sg_index in self.data_info:
    #         for npz_name in self.data_info[sg_index]["npz_names"]:
    #             vqcode_data = {"file_name": npz_name}
    #             npz_path = os.path.join(self.npz_folder, npz_name)
    #             dict_data = np.load(npz_path, allow_pickle=True)
    #             tar_pose_raw = torch.tensor(dict_data["poses"])
    #             tar_trans = torch.tensor(dict_data["trans"]).to(self.rank)
    #             tar_exps = torch.tensor(dict_data["expressions"]).to(self.rank)
    #             tar_pose = tar_pose_raw[:, :165].to(self.rank)
    #             bs, n, j = 1, tar_pose.shape[0], self.joints
    #             data_list.append(tar_pose.detach().cpu().numpy())
    #             data_list_trans.append(tar_trans.detach().cpu().numpy())

    #     data = np.concatenate(data_list, axis=0)
    #     data_trans = np.concatenate(data_list_trans, axis=0)
    #     Mean = data.mean(axis=0)
    #     Std = data.std(axis=0)

    #     np.save(self.mean_seg_path, Mean)
    #     np.save(self.std_seg_path, Std)
    #     if self.use_trans:
    #         trans_mean = data_trans.mean(axis=0)
    #         trans_std = data_trans.std(axis=0)
    #         np.save(self.mean_trans_seg_path, trans_mean)
    #         np.save(self.std_trans_seg_path, trans_std)

    def inv_transform(self, data):
        """The inverse transform of normalization"""

        return data.to(self.device) * torch.from_numpy(self.std).to(self.device) + torch.from_numpy(self.mean).to(self.device)


dict_semantic = {
    "1120": "ARM FLEX",
    "3120": "FOREHEAD SALUTE",
    "3110": "ARM RAISE HIGH-LEVEL",
    "3130": "ARM RAISE MID-LEVEL",
    "1121": "ARM WEIGHTLIFT",
    "2000": "ARMS AKIMBO",
    "2030": "ARMS FOLD",
    "2001": "ARMS RAISE TOWORDS-SKY",
    "3100": "ARMS RAISE V-SHAPE",
    "3111": "ARMS REACH",
    "3330": "HANDS SHELTER",
    "0020": "ARMS SELF-EMBRACE",
    "1122": "ARMS RUN",
    "3101": "ARMS WELCOME",
    "1100": "ARMS EXPLODE",
    "3010": "HANDS RISE",
    "2220": "ARMS DESCEND",
    "3030": "ARMS SPHERICAL",
    "3310": "ARMS FUSE",
    "2002": "ARMS WING",
    "3311": "ARMS SURROUND",
    "0030": "BELLY PAT",
    "0031": "BELLY RUB",
    "0032": "BELLY PREGNANT",
    "0310": "EYEBROW PRESS",
    "0300": "CHEEK BRUSH",
    "0130": "CHEEK SLAP",
    "0320": "CHEEK SUPPORT",
    "1130": "CHEST BEAT",
    "3000": "CHEST HOLD",
    "0200": "CHEST POINT",
    "0301": "CHIN FLICK",
    "0302": "CHIN POINT",
    "0330": "CHIN RUB",
    "0201": "EAR CUP",
    "0220": "EARS BLOCK",
    "0210": "EYE 'TELESCOPE'",
    "0311": "EYE WIPE",
    "1210": "FOREFINGER-AND-MIDDLE-FINGER GAZE",
    "0202": "EYES RING",
    "0131": "FACE COVER",
    "0221": "FACE BARRIER",
    "3320": "FINGERS BECKON",
    "1010": "FINGERS SNAP",
    "2100": "FINGERS SHUT",
    "2330": "FINGERS TALK",
    "3121": "FINGERS WAVE",
    "1000": "FINGERS CROWD-COMPACT",
    "3210": "FINGERTIPS KISS",
    "1330": "FINGERTIPS RUB",
    "1131": "FIST BEAT",
    "1132": "FIST CLENCH",
    "1133": "FIST PUNCH",
    "2221": "ELBOW FALL",
    "1020": "FISTS COMBAT",
    "1110": "FISTS WRING",
    "1111": "FISTS COLLISON",
    "0211": "FOOT TAP",
    "1011": "FOREFINGER BEAT",
    "1230": "FOREFINGER HOP",
    "1211": "FOREFINGER POINT",
    "3131": "FOREFINGER RAISE",
    "1231": "FOREFINGER RAISE-ONE",
    "3102": "FOREFINGER RAISE-SKY",
    "2101": "FOREFINGER WAG",
    "2310": "FOREFINGER SPIN",
    "1232": "FOREFINGER SPIRAL",
    "1030": "FOREFINGER-AND-MIDDLE-FINGER POINT",
    "0000": "FOREFINGER-AND-MIDDLE-FINGER'SMOKE'",
    "1012": "FOREFINGER-AND-MIDDLE-FINGER STAB",
    "3122": "FOREFINGER-AND-MIDDLE-FINGER SALUTE",
    "1031": "FOREFINGER-AND-MIDDLE-FINGER SCISSORS",
    "1220": "FOREFINGER-AND-MIDDLE-FINGER STEPS",
    "2120": "FOREFINGERS AIM",
    "3312": "FOREFINGERS HOOK",
    "1212": "FOREFINGERS POINT-FORWARD",
    "1310": "WRISTS CHANGE",
    "1213": "FOREFINGERS MEASURE",
    "0110": "FOREHEAD SLAP",
    "0111": "FOREHEAD PRESS",
    "0112": "FOREHEAD FINGER-TAP",
    "0132": "FOREHEAD HAND-TAP",
    "0113": "FOREHEAD WIPE",
    "2210": "HAND CHOP",
    "3321": "HAND CALL",
    "3220": "HAND 'DRINK'",
    "0021": "HAND FAN",
    "2110": "HAND FLAP",
    "3331": "HAND FLOP",
    "1013": "HAND JAB",
    "2320": "HAND MEASURE-DOWN",
    "3011": "HAND MEASURE-UP",
    "0010": "HAND PURSE-AROUND",
    "0011": "HAND PURSE",
    "3230": "HAND RING",
    "2311": "HAND ROTATE",
    "1001": "HAND SNATCH",
    "2331": "HAND TOSS",
    "3231": "HAND V-SIGN",
    "2130": "HAND WAG",
    "3123": "HAND WAVE",
    "3300": "HAND 'WRITE'",
    "2300": "PALM HALT",
    "0212": "WRIST CHECK-TIME",
    "2111": "FOREARM THROW-SIDE",
    "2131": "FOREARM REPULSE",
    "2211": "FOREARM CUT",
    "1101": "FIST KNOCK",
    "0001": "TEETH BRUSH",
    "0230": "PALM BARRIER",
    "3332": "HAND SAFEGUARD",
    "2020": "HAND OPEN",
    "1331": "PALM SAW",
    "3020": "HAND DOORWAY-TURN",
    "1311": "HAND HIP-HOP",
    "2132": "HANDS CROSS",
    "1320": "HANDS 'FLUTE'",
    "2200": "HANDS SCISSOR",
    "2021": "HANDS SHRUG",
    "2230": "HANDS 'THROTTLE'",
    "2301": "HANDS T-SIGN",
    "1332": "HANDS APPLAUSE",
    "3021": "HANDS EMPHASIS",
    "1221": "FOREFINGER SCAN",
    "1222": "PALMS TURN-PAGE",
    "2212": "PALMS CROSSCUT",
    "1312": "FINGERS KEYBOARD",
    "2112": "PALMS REPEL",
    "3301": "HANDS EXPLAIN",
    "1300": "HANDS SHOOT",
    "1321": "HANDS PERCUSSION",
    "1322": "HANDS STRUM",
    "1323": "HANDS SERENADE",
    "3211": "HANDS DRAW-BACK",
    "2222": "PALMS OVERTURN",
    "2332": "FOREFINGER EMPTY",
    "3031": "PALMS EXPAND",
    "1301": "HANDS DRAW-OUTLINE",
    "3200": "FINGERS 'LOVE'",
    "1302": "HANDS STEERING",
    "2201": "HANDS DIVIDE",
    "1021": "FIST CLASP",
    "1022": "HANDS BEAST",
    "0002": "HANDS GAME-HANDLE",
    "3022": "HANDS REVEAL",
    "3221": "HANDS FEED",
    "1032": "FINGERS AIR-QUOTES",
    "2010": "HEAD NOD",
    "0331": "HEAD ROLL",
    "0100": "HEAD SCRATCH",
    "2022": "HEAD SHAKE",
    "0303": "HAIR FROOM",
    "0203": "EARS BUNNY",
    "3201": "HEART CLASP",
    "3202": "HEART CROSS",
    "0231": "MOUTH SILENCE",
    "0232": "LIPS ZIP",
    "0120": "MOUTH CLASP",
    "0022": "MOUTH FAN",
    "0321": "MOUTH SHIELD",
    "2231": "NECK CLAMP",
    "0023": "NECK RUB",
    "0332": "NECK SCRATCH",
    "0312": "NOSE FAN",
    "0222": "NOSE TOUCH",
    "2321": "PALM LOWER",
    "1112": "PALM PUNCH",
    "2113": "PALM THRUST",
    "3001": "PALM UP",
    "1102": "PALMS BRUSH",
    "3132": "PALMS CONTACT",
    "2031": "PALMS FRONT",
    "3112": "PALMS UP-HIGH",
    "2312": "PALMS UP-LOW",
    "0012": "PALMS RUB",
    "2313": "PALMS ABSTENTION",
    "2032": "PALMS BOUNDARY",
    "0322": "SHOULDERS SHRUG",
    "0101": "TEMPLE CIRCLE",
    "0121": "TEMPLE 'SHOOT'",
    "0102": "TEMPLE TOUCH",
    "2202": "THROAT 'CUT'",
    "0122": "THROAT GRASP",
    "2121": "THUMB DOWN",
    "3322": "THUMB HITCH",
    "2122": "THUMB POINT",
    "3232": "THUMB UP",
    "2322": "FINGERS ESTIMATE",
    "1200": "THUMB, FOREFINGER AND LITTLE-FINGER RAISE",
    "1002": "WAIST OUTLINE",
    "1201": "FINGERS 'THREE'",
    "1202": "FINGERS 'FIVE'",
    "2023": "PALMS REVERSE",
    "3002": "PALM OFFER",
    "3313": "HANDS UNITE",
    "2232": "HEAD SURRENDER",
    "2302": "ARM ENDEAVOR",
    "3032": "ARMS ENCOMPASS",
    "PALM": "RISE",
    "3222": "HAND TOAST",
}

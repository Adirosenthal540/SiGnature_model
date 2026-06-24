# This code is based on https://github.com/openai/guided-diffusion
import datetime
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from data_loaders.beat2.utils.build_vocab import Vocab
from data_loaders.beat2.data_tools import joints_indexs_names
from data_loaders.beat2.utils import rotation_conversions as rc
from data_loaders.beat2.utils.media import add_audio_to_video
from data_loaders.beat2.utils.other_tools_hf import render_one_sequence_res_npz_only
from data_loaders.get_data import get_dataset_loader
from data_loaders.seg.seg_dataset import SegDataset, find_seg_code_info
from model.mdm import MDM
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import load_model
from utils.parser_util import generate_args
from utils.sampling_utils import unfold_sample_arb_len, double_take_arb_len
import pandas as pd

os.environ["PYOPENGL_PLATFORM"] = "egl"


def calc_frame_colors(handshake_size, blend_size, step_sizes, lengths):
    for ii, step_size in enumerate(step_sizes):
        if ii == 0:
            frame_colors = ["orange"] * (step_size - handshake_size - blend_size) + ["blue"] * blend_size + ["purple"] * (handshake_size // 2)
            continue
        if ii == len(step_sizes) - 1:
            frame_colors += ["purple"] * (handshake_size // 2) + ["blue"] * blend_size + ["orange"] * (lengths[ii] - handshake_size - blend_size)
            continue
        frame_colors += (
            ["purple"] * (handshake_size // 2)
            + ["blue"] * blend_size
            + ["orange"] * (lengths[ii] - 2 * handshake_size - 2 * blend_size)
            + ["blue"] * blend_size
            + ["purple"] * (handshake_size // 2)
        )
    return frame_colors


def configure_environment(seed: int, device: int) -> None:
    """Sets random seed, CUDA device and initializes distributed utilities."""
    fixseed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    dist_util.setup_dist(device)


def prepare_output_directory(args: Any, model_path=None) -> Path:
    """Creates a timestamped output directory based on model path and args."""
    if args.output_dir:
        out_path = Path(args.output_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(args.model_path).parent.name
        iter_label = Path(args.model_path).stem.replace("model", "")
        name_parts = ["DoubleTake_samples", model_dir, iter_label, f"seed{args.seed}"]
        if args.sample_gt:
            name_parts.append("gt")
        name_parts.append(f"handshake{args.handshake_size}")
        if args.double_take:
            name_parts.extend(["double_take", f"blend{args.blend_len}", f"skip{args.skip_steps_double_take}"])
        out_name = "_".join(name_parts + [timestamp, f"diffusion{args.diffusion_steps}"])
        out_path = Path("outputs") / "generated_gestures" / model_dir / out_name

    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def preprocess_text_prompts(args: Any) -> None:
    """Reads .txt or .csv prompt file to set num_samples and batch_size."""
    if not args.input_text:
        return
    prompt_path = Path(args.input_text)
    assert prompt_path.exists(), f"Prompt file not found: {prompt_path}"

    if prompt_path.suffix == ".txt":
        lines = prompt_path.read_text(encoding="utf-8").splitlines()
        args.num_samples = len(lines)
    elif prompt_path.suffix == ".csv":
        df = pd.read_csv(prompt_path)
        args.num_samples = len(df["text"])
    else:
        raise TypeError("Unsupported prompt format: use .txt or .csv")


def load_dataset_and_model(args: Any, dataset_cache_path: str = None) -> Tuple[Any, Any, Any]:
    """Loads test dataset loader and MDM model+diffusion."""
    print("Loading dataset...")
    args.batch_size = 1
    dataset_loader = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size, split="test", device=args.device, shuffle=False, dataset_cache_path=dataset_cache_path
    )
    print("Creating model and diffusion...")
    model, diffusion = load_model(args, dist_util.dev(), ModelClass=MDM)
    model.eval()
    return dataset_loader, model, diffusion


def sample_to_pose_trans(sample, data, device):
    sample = sample.squeeze().T
    sample = data.dataset.inv_transform(sample)

    n_joints: int = 55
    rotations_6d = sample[:, : n_joints * 6]
    translations = sample[:, n_joints * 6 : n_joints * 6 + 3]
    contact = sample[:, -4:]

    n_frames: int = rotations_6d.shape[0]
    rotations_matrix = rc.rotation_6d_to_matrix(rotations_6d.reshape(n_frames, n_joints, 6))
    rotations_angle = rc.matrix_to_axis_angle(rotations_matrix).reshape(n_frames, n_joints * 3)

    pose = rotations_angle.to(device)
    trans = translations.to(device)

    return pose, trans, n_frames


def clear_cache(model):
    for k, v in list(model.get_dict.items()):
        if torch.is_tensor(v):
            v = v.detach()
            if v.is_cuda:
                v = v.cpu()
        model.get_dict[k] = v

    model.get_dict.clear()

    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()


def build_static_pose_gesture_from_json(seg_dataset, device, json_path: str, n_frames: int):
    with open(json_path, "r") as f:
        keep_joints = json.load(f)

    source_motion_name = os.path.basename(json_path).split(".")[0]
    static_pose_npz = os.path.join(os.path.dirname(json_path), f"{source_motion_name}.npz")
    static_pose_frame_index = 196

    joint_names = [joints_indexs_names[i] for i in range(55)]
    edit_bool_per_joint = []
    for j_name in joint_names:
        keep = bool(keep_joints.get(j_name, False))
        edit_bool_per_joint.append(keep)
    joints_rot6d_mask = torch.tensor(np.repeat(np.array(edit_bool_per_joint), 6), dtype=torch.bool)

    static_frame_6d = None

    if static_pose_npz is not None and os.path.exists(static_pose_npz):
        npz = np.load(static_pose_npz, allow_pickle=True)
        if "poses_6d" in npz:
            poses6d = npz["poses_6d"]
            frame_idx = min(max(static_pose_frame_index, 0), poses6d.shape[0] - 1)
            static_frame_6d = poses6d[frame_idx]
        elif "poses" in npz:
            poses_aa = npz["poses"]
            frame_idx = min(max(static_pose_frame_index, 0), poses_aa.shape[0] - 1)
            aa = torch.tensor(poses_aa[frame_idx]).reshape(55, 3)
            rotm = rc.axis_angle_to_matrix(aa.unsqueeze(0))
            sixd = rc.matrix_to_rotation_6d(rotm).reshape(-1).detach().cpu().numpy()
            static_frame_6d = sixd
        else:
            raise ValueError("static_pose_npz must contain 'poses_6d' or 'poses'")
    else:
        raise ValueError("static_pose_npz not found")

    static_frame = np.asarray(static_frame_6d, dtype=np.float32).reshape(1, -1).repeat(n_frames, axis=0)
    static_frame = (static_frame - seg_dataset.mean[:330]) / seg_dataset.std[:330]
    semantic_gesture_info = [
        {"file_name": f"{source_motion_name}.npz", "poses_6d": torch.tensor(static_frame, dtype=torch.float32, device=device).unsqueeze(0)}
    ]

    return {
        "semantic_gesture_index": -1,
        "semantic_gesture_label": f"STATIC_POSE_FROM_{source_motion_name}",
        "semantic_gesture_info": semantic_gesture_info,
        "choice_index": 0,
        "sentence_index": 0,
        "word_index": 0,
        "last_word": "",
        "start_code": 0,
        "end_code": n_frames,
        "joints_rot6d_mask": joints_rot6d_mask,
    }


def main():
    print("Generating samples...")
    args = generate_args()

    configure_environment(args.seed, args.device)

    out_path = prepare_output_directory(args)
    preprocess_text_prompts(args)

    dataset_cache_path = None
    if getattr(args, "person", None):
        person_id = os.path.basename(os.path.dirname(args.model_path)).split("_")[0]
        import yaml
        with open("./dataset/emage.yaml") as f:
            yaml_config = yaml.safe_load(f)
        dataset_cache_path = os.path.join(os.path.dirname(yaml_config["cache_path"]), f"id_{person_id}")

    args.batch_size = 1
    data_loader = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size, split="test", device=args.device,
        shuffle=False, dataset_cache_path=dataset_cache_path,
    )

    print("Creating model and diffusion...")
    model, diffusion = load_model(args, dist_util.dev(), ModelClass=MDM)

    index_person = os.path.basename(data_loader.dataset.args.cache_path).split("_")[-1]
    assert f"{index_person}_" in args.model_path.split("/")[-2], "The model and the dataset doesn't match"
        
    use_seg = args.use_seg
    insert_sg_info_with_a_pose = None
    time_stamps = []

    num_frames_list = []
    if use_seg:
        seg_dataset = SegDataset(
            xlsx_path="./datasets/SeG_SMPLX/SeG_list.xlsx",
            npz_folder="./datasets/SeG_SMPLX/seg_dataset_new_skeleton",
            config_seg_opt_path=args.config_seg_opt_path,
            num_timesteps=diffusion.num_timesteps,
            pose_norm=data_loader.dataset.args.pose_norm,
            mean=data_loader.dataset.mean,
            std=data_loader.dataset.std,
            diffusion=diffusion,
            model=model,
        )
    else:
        seg_dataset = None

    if args.test_data_name is not None and not args.use_seg:
        print("Warning: --test_data_name specified but --use_seg is not enabled. This option only works with segmentation.")

    if args.custom_in_text_semantic is not None and not args.use_seg:
        print("Warning: --custom_in_text_semantic specified but --use_seg is not enabled. This option only works with segmentation.")

    for _, param in model.named_parameters():
        param.requires_grad = False

    found_test_data = False
    processed_count = 0

    for index_batch, (gt_motion, model_kwargs_gt) in enumerate(data_loader):
        if 1:
            sample_gt = gt_motion[0].to(args.device)
            name = model_kwargs_gt["y"]["tar_name"][0][0]
            print(f"name: {name}")

            if args.test_data_name is not None and name != args.test_data_name:
                continue

            found_test_data = True
            processed_count += 1
            print(f"Processing test data: {name}")

            tar_exps = model_kwargs_gt["y"]["tar_exps"][0].to(args.device)
            tar_beta = model_kwargs_gt["y"]["tar_beta"][0].to(args.device)
            tar_id = model_kwargs_gt["y"]["tar_id"][0].to(args.device)

            motions_to_choose_index = None

            if args.custom_in_text_semantic is not None:
                print(f"Origin Text: {model_kwargs_gt["y"]["in_text"]}")

                in_text_semantic = args.custom_in_text_semantic

                print(f"LLM tag Text: {model_kwargs_gt["y"]["in_text_semantic"]}")
                print(f"Tagged Text: {in_text_semantic}")

                print(f"Using custom in_text_semantic: {in_text_semantic}")
            else:
                in_text_semantic = model_kwargs_gt["y"]["in_text_semantic"]

            in_word = model_kwargs_gt["y"]["tokens"][0].to(args.device)
            tar_pose, tar_trans, n_frames = sample_to_pose_trans(sample_gt, data_loader, args.device)

            if use_seg:
                if in_text_semantic[0] == "":
                    continue
                insert_sg_info_with_a_pose = find_seg_code_info(
                    in_word.cpu().numpy(),
                    data_loader.dataset.lang_model,
                    seg_dataset,
                    in_text_semantic
                )
                if getattr(args, "static_pose_json", None):
                    static_gesture = build_static_pose_gesture_from_json(seg_dataset, args.device, args.static_pose_json, n_frames)
                    static_gesture["start_code"] = 0
                    static_gesture["end_code"] = n_frames
                    static_gesture["static_pose"] = True
                    static_gesture["lambda_integration"] = 1
                    static_gesture["blend_frames"] = 0
                    static_gesture["use_global_pose"] = True
                    insert_sg_info_with_a_pose = (insert_sg_info_with_a_pose or []) + [static_gesture]

            n = tar_pose.shape[0]

            pre_frames = args.handshake_size
            roundt = (n) // (data_loader.dataset.args.pose_length - pre_frames) + 1
            remain = (n) % (data_loader.dataset.args.pose_length - pre_frames)
            round_l = data_loader.dataset.args.pose_length - pre_frames

            audio_fps = data_loader.dataset.args.audio_fps
            pose_fps = data_loader.dataset.args.pose_fps
            audio_data = model_kwargs_gt["y"]["in_audio_resample"][0]
            round_model_kwargs = {
                "y": {
                    "tar_beta": [],
                    "tar_id": [],
                    "tokens": [],
                    "text": [],
                    "audio": [],
                    "lengths": [],
                    "frames_indexs": [],
                }
            }
            tokens_orig = model_kwargs_gt["y"]["tokens"][0]
            audio_orig = model_kwargs_gt["y"]["audio"][0]

            min_arb_len = args.handshake_size + args.blend_len + 10
            max_arb_len = data_loader.dataset.args.pose_length

            for i in range(0, roundt):
                if i == roundt - 1 and remain < min_arb_len:
                    roundt = roundt - 1
                    n = n - remain
                    continue
                elif (i == roundt - 1 and remain >= min_arb_len) or (i == roundt - 2 and remain < min_arb_len):
                    in_word_tmp = torch.zeros(max_arb_len).to(torch.long)
                    in_word_tmp[: len(tokens_orig[i * (round_l) : (i + 1) * (round_l) + pre_frames])] = tokens_orig[
                        i * (round_l) : (i + 1) * (round_l) + pre_frames
                    ]
                    round_model_kwargs["y"]["tokens"].append(in_word_tmp)
                    round_model_kwargs["y"]["frames_indexs"].append([i * (round_l), (i + 1) * (round_l) + pre_frames])
                    round_model_kwargs["y"]["text"].append(data_loader.dataset._create_text_from_in_word(in_word_tmp))
                    round_model_kwargs["y"]["tar_beta"].append(model_kwargs_gt["y"]["tar_beta"][0][: (round_l + pre_frames)])
                    round_model_kwargs["y"]["tar_id"].append(model_kwargs_gt["y"]["tar_id"][0][: (round_l + pre_frames)])
                    in_audio_tmp = torch.zeros_like(round_model_kwargs["y"]["audio"][-1])
                    in_audio_tmp[
                        : len(
                            audio_orig[
                                i * (audio_fps // pose_fps * round_l) : (i + 1) * (audio_fps // pose_fps * round_l)
                                + audio_fps // pose_fps * pre_frames
                            ]
                        ),
                        :,
                    ] = audio_orig[
                        i * (audio_fps // pose_fps * round_l) : (i + 1) * (audio_fps // pose_fps * round_l) + audio_fps // pose_fps * pre_frames
                    ]
                    round_model_kwargs["y"]["audio"].append(in_audio_tmp)
                    round_model_kwargs["y"]["lengths"].append(len(tokens_orig[i * (round_l) : (i + 1) * (round_l) + pre_frames]))

                else:
                    in_word_tmp = tokens_orig[i * (round_l) : (i + 1) * (round_l) + pre_frames]
                    round_model_kwargs["y"]["tokens"].append(in_word_tmp)
                    round_model_kwargs["y"]["frames_indexs"].append([i * (round_l), (i + 1) * (round_l) + pre_frames])
                    round_model_kwargs["y"]["text"].append(data_loader.dataset._create_text_from_in_word(in_word_tmp))
                    round_model_kwargs["y"]["tar_beta"].append(model_kwargs_gt["y"]["tar_beta"][0][: (round_l + pre_frames)])
                    round_model_kwargs["y"]["tar_id"].append(model_kwargs_gt["y"]["tar_id"][0][: (round_l + pre_frames)])
                    in_audio_tmp = audio_orig[
                        i * (audio_fps // pose_fps * round_l) : (i + 1) * (audio_fps // pose_fps * round_l) + audio_fps // pose_fps * pre_frames
                    ]
                    round_model_kwargs["y"]["audio"].append(in_audio_tmp)
                    round_model_kwargs["y"]["lengths"].append(data_loader.dataset.args.pose_length)

            round_model_kwargs["y"]["lengths"] = torch.tensor(round_model_kwargs["y"]["lengths"], device=dist_util.dev()).int()

            tar_poses, tar_transs, n_framess = [], [], []
            for rep_i in range(args.batch_size):
                tar_pose, tar_trans, n_frames = sample_to_pose_trans(gt_motion[rep_i], data_loader, args.device)
                tar_poses.append(tar_pose)
                tar_transs.append(tar_trans)
                n_framess.append(n_frames)

            all_motions = []
            all_motions_seg = []
            all_lengths = []
            all_text = []
            all_captions = []
            model_kwargs = round_model_kwargs
            for rep_i in range(args.num_repetitions):
                print(f"### Sampling [repetitions #{rep_i}]")
                if args.guidance_param != 1:
                    model_kwargs["y"]["scale"] = torch.ones(roundt, device=dist_util.dev()) * args.guidance_param
                model_kwargs["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs["y"].items()}

                step_sizes = np.zeros(len(model_kwargs["y"]["lengths"]), dtype=int)
                for ii, len_i in enumerate(model_kwargs["y"]["lengths"]):
                    if ii == 0:
                        step_sizes[ii] = len_i
                        continue
                    step_sizes[ii] = step_sizes[ii - 1] + len_i - args.handshake_size
                final_n_frames = step_sizes[-1]
                if use_seg:
                    insert_sg_info_with_a_pose_data = {}
                    insert_sg_info_with_a_pose_data["gestures"] = insert_sg_info_with_a_pose
                    insert_sg_info_with_a_pose_data["action"] = "edit"
                    samples_per_rep_list_with_seg, samples_type_with_seg = double_take_arb_len(
                        args,
                        diffusion,
                        model,
                        model_kwargs,
                        max_arb_len,
                        seg_dataset=seg_dataset,
                        insert_sg_info_with_a_pose=insert_sg_info_with_a_pose_data,
                        data_loader=data_loader,
                    )
                    for sample_i_seg, samples_type_i_seg in zip(samples_per_rep_list_with_seg, samples_type_with_seg):
                        sample_seg = unfold_sample_arb_len(sample_i_seg, args.handshake_size, step_sizes, final_n_frames, model_kwargs)
                        all_motions_seg.append(sample_seg.cpu().numpy())
                    all_motions_seg = np.concatenate(all_motions_seg, axis=0)

                if seg_dataset:
                    seg_dataset.integration_case = 0
                    insert_sg_info_with_a_pose_data["action"] = "transfer"
                    samples_per_rep_list, samples_type = double_take_arb_len(
                        args,
                        diffusion,
                        model,
                        model_kwargs,
                        max_arb_len,
                        seg_dataset=seg_dataset,
                        insert_sg_info_with_a_pose=insert_sg_info_with_a_pose_data,
                        data_loader=data_loader,
                    )
                    for sample_i, samples_type_i in zip(samples_per_rep_list, samples_type):
                        sample = unfold_sample_arb_len(sample_i, args.handshake_size, step_sizes, final_n_frames, model_kwargs)
                        all_motions.append(sample.cpu().numpy())
                        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())
                    all_motions = np.concatenate(all_motions, axis=0)
                    use_base = True
                    if use_base:
                        all_motions_base = []
                        samples_per_rep_list_base, samples_type_base = double_take_arb_len(
                            args, diffusion, model, model_kwargs, max_arb_len
                        )
                        for sample_i, samples_type_i in zip(samples_per_rep_list_base, samples_type_base):
                            sample_base = unfold_sample_arb_len(sample_i, args.handshake_size, step_sizes, final_n_frames, model_kwargs)
                            all_motions_base.append(sample_base.cpu().numpy())
                        all_motions_base = np.concatenate(all_motions_base, axis=0)

                else:
                    time_stamp = time.time()
                    samples_per_rep_list, samples_type = double_take_arb_len(args, diffusion, model, model_kwargs, max_arb_len)

                    for sample_i, samples_type_i in zip(samples_per_rep_list, samples_type):
                        sample = unfold_sample_arb_len(sample_i, args.handshake_size, step_sizes, final_n_frames, model_kwargs)
                        all_motions.append(sample.cpu().numpy())
                        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())
                    all_motions = np.concatenate(all_motions, axis=0)
                    time_stamp = time.time() - time_stamp
                    time_stamps.append(time_stamp)
                    num_frames_list.append(all_motions.shape[-1])
                    print(f"Time taken: {time_stamp} seconds")

            args.num_samples = 1
            args.batch_size = 1
            n_frames = final_n_frames

            num_repetitions = 1
            all_lengths = [n_frames] * num_repetitions

            out_path = str(out_path)

            print(f"saving visualizations to [{out_path}]...")

            all_text_new = []
            time_of_caption = 30
            round_t = (n) // (time_of_caption)
            for i in range(0, round_t):
                in_word_tmp = tokens_orig[i * (time_of_caption) : (i + 1) * (time_of_caption)]
                all_text_new.append(data_loader.dataset._create_text_from_in_word(in_word_tmp))

            for sample_i in range(args.num_samples):
                for rep_i, samples_type_i in zip(range(num_repetitions), samples_type):
                    caption = [f"{samples_type_i} {all_text_new[0]}"] * time_of_caption
                    for ii in range(1, round_t):
                        caption += [f"{samples_type_i} {all_text_new[ii]}"] * time_of_caption
                    length = all_lengths[rep_i * args.batch_size + sample_i]
                    motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]

                    if use_seg:
                        base_motion = all_motions_base[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                        motion_seg = all_motions_seg[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                        rec_pose, rec_trans, n_frames = sample_to_pose_trans(torch.tensor(motion_seg).permute(1, 2, 0), data_loader, args.device)
                        tar_pose, tar_trans, n_frames = sample_to_pose_trans(sample_gt, data_loader, args.device)
                        base_pose_base, base_trans_base, n_frames = sample_to_pose_trans(
                            torch.tensor(base_motion).permute(1, 2, 0), data_loader, args.device
                        )

                        banchmark_pose, banchmark_trans, n_frames = sample_to_pose_trans(
                            torch.tensor(motion).permute(1, 2, 0), data_loader, args.device
                        )

                        tar_trans_np = tar_trans.detach().cpu().numpy().reshape(-1, 3)[:n_frames]
                    else:
                        rec_pose, rec_trans, n_frames = sample_to_pose_trans(torch.tensor(motion).permute(1, 2, 0), data_loader, args.device)
                        tar_trans_np = tar_trans.detach().cpu().numpy().reshape(-1, 3)[:n_frames]

                    n_frames = min(rec_trans.shape[0], args.batch_size * n_frames)
                    tar_pose_np = tar_pose.detach().cpu().numpy()[:n_frames]
                    rec_pose_np = rec_pose.detach().cpu().numpy()[:n_frames]
                    rec_trans_np = rec_trans.detach().cpu().numpy().reshape(args.batch_size * n_frames, 3)
                    rec_exp_np = tar_exps.detach().cpu().numpy()[:n_frames].reshape(args.batch_size * (n_frames), 100)
                    tar_exp_np = tar_exps.detach().cpu().numpy()[:n_frames].reshape(args.batch_size * (n_frames), 100)

                    tar_beta_np = tar_beta[0].detach().cpu().numpy()

                    year_smplx = data_loader.dataset.loaded_args["smplx_params"]["gender"].split("_")[-1]

                    if use_seg:
                        benchmark_pose_np = banchmark_pose.detach().cpu().numpy()[:n_frames]
                        benchmark_trans_np = banchmark_trans.detach().cpu().numpy().reshape(args.batch_size * n_frames, 3)
                        benchmark_path: str = os.path.join(out_path, f"benchmark_{name}.npz")
                        benchmark_mean_trans = np.mean(benchmark_trans_np, axis=0)
                        benchmark_mean_trans[1] = 0
                        np.savez(
                            benchmark_path,
                            betas=tar_beta_np,
                            poses=benchmark_pose_np,
                            expressions=tar_exp_np,
                            trans=benchmark_trans_np - benchmark_mean_trans,
                            model=f"smpx{year_smplx}",
                            gender="neutral",
                            mocap_frame_rate=data_loader.dataset.loaded_args["render_video_fps"],
                        )

                        base_pose_np = base_pose_base.detach().cpu().numpy()[:n_frames]
                        base_trans_np = base_trans_base.detach().cpu().numpy().reshape(args.batch_size * n_frames, 3)
                        base_path: str = os.path.join(out_path, f"base_{name}.npz")
                        base_mean_trans = np.mean(base_trans_np, axis=0)
                        base_mean_trans[1] = 0
                        np.savez(
                            base_path,
                            betas=tar_beta_np,
                            poses=base_pose_np,
                            expressions=tar_exp_np,
                            trans=base_trans_np - base_mean_trans,
                            model=f"smpx{year_smplx}",
                            gender="neutral",
                            mocap_frame_rate=data_loader.dataset.loaded_args["render_video_fps"],
                        )

                    gt_path: str = os.path.join(out_path, f"gt_{name}.npz")
                    tar_mean_trans = np.mean(tar_trans_np, axis=0)
                    tar_mean_trans[1] = 0
                    np.savez(
                        gt_path,
                        betas=tar_beta_np,
                        poses=tar_pose_np,
                        expressions=tar_exp_np,
                        trans=tar_trans_np - tar_mean_trans,
                        model=f"smplx{year_smplx}",
                        gender="neutral",
                        mocap_frame_rate=data_loader.dataset.loaded_args["render_video_fps"],
                    )

                    res_path: str = os.path.join(out_path, f"res_{name}.npz")
                    rec_mean_trans = np.mean(rec_trans_np, axis=0)
                    rec_mean_trans[1] = 0
                    np.savez(
                        res_path,
                        betas=tar_beta_np,
                        poses=rec_pose_np,
                        expressions=rec_exp_np,
                        trans=rec_trans_np - rec_mean_trans,
                        model=f"smplx{year_smplx}",
                        gender="neutral",
                        mocap_frame_rate=data_loader.dataset.loaded_args["render_video_fps"],
                    )
                    if args.run_videos:
                        render_args = data_loader.dataset.args
                        smplx_params = data_loader.dataset.loaded_args["smplx_params"]

                        if use_seg:
                            npz_renders = [
                                (benchmark_path, f"benchmark_{name}"),
                                (res_path, f"res_{name}"),
                                (base_path, f"base_{name}"),
                            ]
                        else:
                            npz_renders = [
                                (res_path, f"res_{name}"),
                                (gt_path, f"gt_{name}"),
                            ]

                        audio_sr = data_loader.dataset.args.audio_sr
                        audio_data_trimmed = audio_data[: int(np.floor(n / pose_fps * audio_sr))]

                        for npz_path, render_name in npz_renders:
                            silent_path = render_one_sequence_res_npz_only(
                                res_npz_path=npz_path,
                                output_dir=out_path,
                                args=render_args,
                                smplx_params=smplx_params,
                                device=args.device,
                                name=render_name,
                            )
                            final_path = os.path.join(out_path, f"{render_name}.mp4")
                            add_audio_to_video(silent_path, final_path, audio_data_trimmed, audio_sr)
                            os.remove(silent_path)
                            print(final_path)

            abs_path = os.path.abspath(out_path)
            print(f"[Done] Results are at [{abs_path}]")
    print(f"Time taken: {time_stamps} seconds")
    print(f"num frames: {np.sum(num_frames_list)}")
    print(f"Average time taken: {np.mean(time_stamps)} seconds")
    print(f"sum time taken: {np.sum(time_stamps) } seconds")
    if args.test_data_name is not None and not found_test_data:
        print(f"Error: Test data '{args.test_data_name}' not found in the dataset.")
        print("Available test data names can be found by running without --test_data_name")
    elif args.test_data_name is not None:
        print(f"Successfully processed {processed_count} item(s) for test data: {args.test_data_name}")


if __name__ == "__main__":
    main()

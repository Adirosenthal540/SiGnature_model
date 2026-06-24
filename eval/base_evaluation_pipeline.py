import shutil
import matplotlib.pyplot as plt
import glob
from sklearn import metrics
from data_loaders.beat2 import data_tools
from utils.model_util import load_model
from utils.parser_util import evaluate_args, evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from tqdm import tqdm
from collections import OrderedDict
from data_loaders.beat2.utils.build_vocab import Vocab
from data_loaders.beat2.utils import config, other_tools, metric
import torch.distributed as dist
import os
import json
from pathlib import Path
from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
import librosa
import utils.rotation_conversions as rc
import numpy as np
import torch
import pandas as pd
from data_loaders.beat2.utils.other_tools_hf import (
    render_one_sequence,
    postprocess_silent_video,
)

# Set up headless rendering for pyrender
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["DISPLAY"] = ":99"  # Fallback display
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"

torch.multiprocessing.set_sharing_strategy("file_system")


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print("========== Evaluating Matching Score ==========")
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):

                motion, cond = batch

                (
                    masks,
                    lengths,
                    texts,
                    tokens,
                    tar_trans,
                    tar_exps,
                    tar_beta,
                    tar_pose,
                ) = (
                    cond["y"]["mask"],
                    cond["y"]["lengths"],
                    cond["y"]["text"],
                    cond["y"]["tokens"],
                    cond["y"]["tar_trans"],
                    cond["y"]["tar_exps"],
                    cond["y"]["tar_beta"],
                    cond["y"]["tar_pose"],
                )

                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch  # todo
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens,
                )

                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += len(masks)  # num of batch size (i think) #text_embeddings.shape[0]

                # all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}")
        print(
            f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}",
            file=file,
            flush=True,
        )

        line = f"---> [{motion_loader_name}] R_precision: "
        for i in range(len(R_precision)):
            line += "(top %d): %.4f " % (i + 1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print("========== Evaluating FID ==========")
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(motions=motions, m_lens=m_lens)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f"---> [{model_name}] FID: {fid:.4f}")
        print(f"---> [{model_name}] FID: {fid:.4f}", file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print("========== Evaluating Diversity ==========")
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f"---> [{model_name}] Diversity: {diversity:.4f}")
        print(f"---> [{model_name}] Diversity: {diversity:.4f}", file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print("========== Evaluating MultiModality ==========")
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f"---> [{model_name}] Multimodality: {multimodality:.4f}")
        print(
            f"---> [{model_name}] Multimodality: {multimodality:.4f}",
            file=file,
            flush=True,
        )
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(
    eval_wrapper,
    gt_loader,
    eval_motion_loaders,
    log_file,
    replication_times,
    diversity_times,
    mm_num_times,
    run_mm=False,
):
    with open(log_file, "w") as f:
        all_metrics = OrderedDict(
            {
                "Matching Score": OrderedDict({}),
                "R_precision": OrderedDict({}),
                "FID": OrderedDict({}),
                "Diversity": OrderedDict({}),
                "MultiModality": OrderedDict({}),
            }
        )
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders["ground truth"] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f"==================== Replication {replication} ====================")
            print(
                f"==================== Replication {replication} ====================",
                file=f,
                flush=True,
            )
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f"Time: {datetime.now()}")
                print(f"Time: {datetime.now()}", file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            print(f"!!! DONE !!!")
            print(f"!!! DONE !!!", file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics["Matching Score"]:
                    all_metrics["Matching Score"][key] = [item]
                else:
                    all_metrics["Matching Score"][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics["R_precision"]:
                    all_metrics["R_precision"][key] = [item]
                else:
                    all_metrics["R_precision"][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics["FID"]:
                    all_metrics["FID"][key] = [item]
                else:
                    all_metrics["FID"][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics["Diversity"]:
                    all_metrics["Diversity"][key] = [item]
                else:
                    all_metrics["Diversity"][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics["MultiModality"]:
                        all_metrics["MultiModality"][key] = [item]
                    else:
                        all_metrics["MultiModality"][key] += [item]

        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print("========== %s Summary ==========" % metric_name)
            print("========== %s Summary ==========" % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + "_" + model_name] = mean
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}")
                    print(
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}",
                        file=f,
                        flush=True,
                    )
                elif isinstance(mean, np.ndarray):
                    line = f"---> [{model_name}]"
                    for i in range(len(mean)):
                        line += "(top %d) Mean: %.4f CInt: %.4f;" % (
                            i + 1,
                            mean[i],
                            conf_interval[i],
                        )
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


def load_evaluation_model(args, device=0):
    eval_model_module = __import__(f"data_loaders.beat2.models.{args.eval_model}", fromlist=["something"])
    # eval copy is for single card evaluation
    if args.ddp:
        eval_model = getattr(eval_model_module, args.e_name)(args).to(device)
    else:
        eval_model = getattr(eval_model_module, args.e_name)(args).to(device)

    other_tools.load_checkpoints(eval_model, args.data_path + args.e_path, device, args.e_name)

    if args.ddp:
        eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(eval_model, process_group)
        eval_model = DDP(
            eval_model,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    eval_model.eval()

    if device == 0:
        # logger.info(eval_model)
        logger.info(f"init {args.e_name} success")
        if args.stat == "wandb":
            wandb.watch(eval_model)
    return eval_model


def calculate_speed_3d(positions, time_interval):
    # positions: numpy array of shape (bs, n_samples, 3) where each row is [x, y, z]
    # time_interval: time interval between position samples

    # Calculate velocities
    velocities = np.gradient(positions, time_interval, axis=1)

    # Calculate accelerations
    accelerations = np.gradient(velocities, time_interval, axis=1)

    # Calculate jerk
    jerks = np.gradient(accelerations, time_interval, axis=1)

    # Average jerk over time and dimensions
    avg_velocities = np.mean(np.linalg.norm(velocities, axis=2))
    avg_accelerations = np.mean(np.linalg.norm(accelerations, axis=2))
    avg_jerk = np.mean(np.linalg.norm(jerks, axis=2))

    return avg_velocities, avg_accelerations, avg_jerk


def calculate_mean_absolute_jerk(positions, time_interval):
    """
    Mean Absolute Jerk (MAJ): average magnitude of the third derivative of position.
    Lower values indicate smoother motion.
    """
    velocities = np.gradient(positions, time_interval, axis=1)
    accelerations = np.gradient(velocities, time_interval, axis=1)
    jerks = np.gradient(accelerations, time_interval, axis=1)
    return float(np.mean(np.linalg.norm(jerks, axis=2)))


def calculate_acceleration_continuity(positions, time_interval):
    """
    Acceleration Continuity (AccC): mean absolute frame-to-frame change in acceleration magnitude.
    Lower values indicate more temporally consistent acceleration.
    """
    velocities = np.gradient(positions, time_interval, axis=1)
    accelerations = np.gradient(velocities, time_interval, axis=1)
    accel_mag = np.linalg.norm(accelerations, axis=2)
    accel_mag_diff = np.diff(accel_mag, axis=1)
    return float(np.mean(np.abs(accel_mag_diff)))


def save_motion_as_npz(motion_data, poses, translations, betas, expressions, output_path):
    """
    Save motion data as NPZ file for video rendering.
    
    Args:
        motion_data: Original motion tensor
        poses: Generated poses
        translations: Generated translations
        betas: Beta parameters
        expressions: Expression parameters
        output_path: Path to save NPZ file
    """
    np.savez(
        output_path,
        poses=poses.cpu().numpy(),
        trans=translations.cpu().numpy(),
        betas=betas.cpu().numpy(),
        expression=expressions.cpu().numpy(),
        motion=motion_data.cpu().numpy() if motion_data is not None else None,
    )


def generate_video_comparison(
    gt_motion, gt_poses, gt_trans, gt_betas, gt_expressions,
    rec_motion, rec_poses, rec_trans, rec_betas, rec_expressions,
    output_dir, person_name, batch_idx, sample_idx, device, dataset_args, smplx_params,
    plot_video=True, audio_data=None, text_data=None
):
    """
    Generate video comparison between GT and generated motion.
    
    Args:
        gt_motion: Ground truth motion tensor
        gt_poses: Ground truth poses
        gt_trans: Ground truth translations
        gt_betas: Ground truth betas
        gt_expressions: Ground truth expressions
        rec_motion: Generated motion tensor
        rec_poses: Generated poses
        rec_trans: Generated translations
        rec_betas: Generated betas
        rec_expressions: Generated expressions
        output_dir: Directory to save videos
        person_name: Name of the person
        batch_idx: Batch index
        sample_idx: Sample index
        device: Device to use
        dataset_args: Dataset arguments
        smplx_params: SMPLX parameters
        plot_video: Whether to generate video
        audio_data: Audio data for the video
        text_data: Text data for the video
    """
    if not plot_video:
        return

    try:
        # Create debug directory
        debug_dir = Path(output_dir) / "debug"
        debug_dir.mkdir(exist_ok=True)

        # Save GT motion as NPZ
        gt_npz_path = debug_dir / f"{person_name}_batch_{batch_idx}_sample_{sample_idx}_gt.npz"
        # np.savez(gt_npz_path, gt_poses.detach().cpu().numpy(), gt_trans.detach().cpu().numpy(), gt_betas.detach().cpu().numpy(), gt_expressions.detach().cpu().numpy())
        gt_trans = gt_trans.detach().cpu().numpy()
        gt_poses = gt_poses.detach().cpu().numpy()
        gt_betas = gt_betas.detach().cpu().numpy()
        gt_expressions = gt_expressions.detach().cpu().numpy()
        # betas = gt_betas[0]

        gt_trans_mean = np.mean(gt_trans, axis=0)
        gt_trans_mean[1] = 0
        np.savez(
            gt_npz_path,
            betas=gt_betas,
            poses=gt_poses,
            expressions=gt_expressions,
            trans=gt_trans - gt_trans_mean,
            model="smplx2020",
            gender="neutral",
            mocap_frame_rate=30,
        )

        # Save generated motion as NPZ
        rec_npz_path = debug_dir / f"{person_name}_batch_{batch_idx}_sample_{sample_idx}_rec.npz"
        # np.savez(rec_npz_path, rec_poses.detach().cpu().numpy(), rec_trans.detach().cpu().numpy(), rec_betas.detach().cpu().numpy(), rec_expressions.detach().cpu().numpy())
        rec_trans = rec_trans.detach().cpu().numpy()
        rec_poses = rec_poses.detach().cpu().numpy()
        rec_betas = rec_betas.detach().cpu().numpy()
        rec_expressions = rec_expressions.detach().cpu().numpy()
        # betas = rec_betas[0]

        rec_trans_mean = np.mean(rec_trans, axis=0)
        rec_trans_mean[1] = 0
        np.savez(
            rec_npz_path,
            betas=rec_betas,
            poses=rec_poses,
            expressions=rec_expressions,
            trans=rec_trans - rec_trans_mean,
            model="smplx2020",
            gender="neutral",
            mocap_frame_rate=30,
        )

        # Generate 2-panel video: GT vs Generated
        video_path = render_one_sequence(
            res_npz_path=rec_npz_path,
            gt_npz_path=gt_npz_path,
            output_dir=output_dir,
            smplx_params=smplx_params,
            args=dataset_args,
            device=device,
            name=f"{person_name}_batch_{batch_idx}_sample_{sample_idx}_comparison",
            titles=["Generated", "GT"],
            overlay=False,
        )

        # Add audio to the video if available
        if audio_data is not None and text_data is not None:
            try:
                postprocess_silent_video(video_path, text_data, audio_data)
                print(f"✅ Added audio to video: {video_path}")
            except Exception as e:
                print(f"⚠️ Failed to add audio to video: {e}")
        else:
            print(f"✅ Generated video: {video_path}")

    except Exception as e:
        print(f"⚠️ Failed to generate video: {e}")


def run_multi_checkpoint_evaluation(args, validated_mapping_json, plot_video=False, load_generated_motion=None):
    """
    Run evaluation on multiple checkpoints using a JSON mapping file.

    Args:
        args: Evaluation arguments
        validated_mapping_json: Path to JSON file containing validated mapping
        plot_video: Whether to generate comparison videos
        load_generated_motion: Path to NPZ file containing pre-generated motion
    """
    print(f"🔄 Starting multi-checkpoint evaluation using: {validated_mapping_json}")

    # Load validated mapping
    try:
        with open(validated_mapping_json, "r") as f:
            validated_mapping = json.load(f)
        print(f"📋 Loaded validated mapping with {len(validated_mapping)} persons")
    except Exception as e:
        print(f"❌ Error loading validated mapping JSON: {e}")
        return

    # Set fps and number of frames to generate
    fps: float = 30
    max_frames: int = 196
    n_frames: int = min(max_frames, int(args.motion_length * fps))

    # Set Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device: int = args.device

    # Create evaluation folder
    evaluation_folder = f"./evaluation_results/multi_checkpoint_base_model_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(evaluation_folder, exist_ok=True)

    # Create folder for specific timestamp
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(evaluation_folder, f"{timestamp}_seed_{args.seed}")
    os.makedirs(out_path, exist_ok=True)

    logger.configure(dir=out_path)

    # Initialize results storage
    all_results = {}
    cross_person_fid_data = {}

    # Initialize evaluation model
    emage_args = config.parse_args()
    eval_model = load_evaluation_model(emage_args, device)
    eval_model.to(device)
    eval_model.eval()

    batch_size = 1
    n_joints = 55
    align_mask = 0

    # Process each person's checkpoint
    for person_name, (checkpoint_path, dataset_cache_path) in validated_mapping.items():
        print(f"\n{'='*60}")
        print(f"🎯 Evaluating person: {person_name}")
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"📁 Dataset cache: {dataset_cache_path}")
        print(f"{'='*60}")

        # try:
        if 1:
            # Load dataset with specific cache path
            eval_dataloader = get_dataset_loader(
                name=args.dataset,
                batch_size=batch_size,
                split="val",
                device=device,
                dataset_cache_path=dataset_cache_path,
            )

            # Load model
            logger.info("Creating model and diffusion...")
            model, diffusion = load_model(args, dist_util.dev(), model_path=checkpoint_path)

            # Initialize metrics for this person
            l1_calculator = metric.L1div()
            align = 0
            total_length = 0
            latent_out = []
            latent_ori = []
            jerk_vel = []
            jerk_vel_diff = []
            avg_velocities_diff = []
            avg_accelerations_diff = []
            maj_values = []
            maj_diffs = []
            acc_cont_values = []
            acc_cont_diffs = []

            # Initialize cross-person FID data
            cross_person_fid_data[person_name] = {"model_results": [], "gt_results": []}

            # Initialize alignmenter
            alignmenter = metric.alignment(0.3, 7, eval_dataloader.dataset.avg_vel, upper_body=[3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])

            # Process batches
            with torch.no_grad():
                for index_batch, (motion, cond) in tqdm(enumerate(eval_dataloader), desc=f"Processing {person_name}"):
                    print(f"index_batch {index_batch} / {len(eval_dataloader)}")

                    motion = motion.to(device)
                    bs, n, j = motion.shape[0], motion.shape[-1], n_joints

                    cond["y"] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond["y"].items()}
                    tar_exps = torch.stack(cond["y"]["tar_exps"]).to(device)
                    tar_beta = torch.stack(cond["y"]["tar_beta"]).to(device)
                    tar_pose = torch.stack(cond["y"]["tar_pose"]).to(device)
                    tar_trans = torch.stack(cond["y"]["tar_trans"]).to(device)

                    avg_velocities_gt, avg_accelerations_gt, jerk_vel_gt_motions = calculate_speed_3d(tar_trans.cpu().numpy(), 1 / fps)
                    maj_gt = calculate_mean_absolute_jerk(tar_trans.cpu().numpy(), 1 / fps)
                    acc_cont_gt = calculate_acceleration_continuity(tar_trans.cpu().numpy(), 1 / fps)
                    tar_pose_mat = rc.axis_angle_to_matrix(tar_pose.reshape(bs * n, j, 3))
                    tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_mat).reshape(bs, n, j * 6)

                    remain = n % emage_args.vae_test_len

                    # Store GT latent for cross-person FID
                    latent_ori.append(eval_model.map2latent(tar_pose_6d[:, : n - remain]).reshape(-1, emage_args.vae_length).cpu().numpy())

                    # Generate sample
                    sample_fn = diffusion.p_sample_loop
                    sample = sample_fn(
                        model,
                        (batch_size, model.njoints, model.nfeats, max_frames),
                        clip_denoised=False,
                        model_kwargs=cond,
                        skip_timesteps=0,
                        init_image=None,
                        progress=True,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                    )

                    # Process sample
                    sample = sample.squeeze(dim=2).permute(0, 2, 1)
                    sample = eval_dataloader.dataset.inv_transform(sample)

                    # Split network output
                    rotations_6d = sample[:, :, : n_joints * 6]
                    translations = sample[:, :, n_joints * 6 : n_joints * 6 + 3]

                    # Convert to pose
                    n_frames = rotations_6d.shape[1]
                    rotations_matrix = rc.rotation_6d_to_matrix(rotations_6d.reshape(bs, n_frames, n_joints, 6))
                    rotations_angle = rc.matrix_to_axis_angle(rotations_matrix).reshape(bs, n_frames, n_joints * 3)

                    rec_pose = rotations_angle.to(device)
                    rec_trans = translations.to(device)
                    avg_velocities_rec, avg_accelerations_rec, jerk_vel_motions = calculate_speed_3d(translations.cpu().numpy(), 1 / fps)

                    # Store metrics
                    jerk_vel.append(jerk_vel_motions)
                    avg_velocities_diff.append(np.abs(avg_velocities_gt - avg_velocities_rec))
                    avg_accelerations_diff.append(np.abs(avg_accelerations_gt - avg_accelerations_rec))
                    jerk_vel_diff.append(np.abs(jerk_vel_gt_motions - jerk_vel_motions))

                    # Store model latent for cross-person FID
                    latent_out.append(
                        eval_model.map2latent(rotations_6d.reshape(bs, n, -1)[:, : n - remain].to(device))
                        .reshape(-1, emage_args.vae_length)
                        .cpu()
                        .numpy()
                    )

                    # Process each sample in batch
                    for indx_samp in range(bs):
                        # L1 Diversity calculation
                        smplx_output_rec = eval_dataloader.dataset.smplx(
                            betas=tar_beta[indx_samp, :n_frames, :],
                            transl=rec_trans[indx_samp] - rec_trans[indx_samp],
                            expression=tar_exps[indx_samp, :n_frames, :] - tar_exps[indx_samp, :n_frames, :],
                            jaw_pose=rec_pose[indx_samp, :, 66:69],
                            global_orient=rec_pose[indx_samp, :, :3],
                            body_pose=rec_pose[indx_samp, :, 3 : 21 * 3 + 3],
                            left_hand_pose=rec_pose[indx_samp, :, 25 * 3 : 40 * 3],
                            right_hand_pose=rec_pose[indx_samp, :, 40 * 3 : 55 * 3],
                            return_joints=True,
                            leye_pose=rec_pose[indx_samp, :, 69:72],
                            reye_pose=rec_pose[indx_samp, :, 72:75],
                            return_verts=False,
                        )

                        joints_rec = smplx_output_rec.joints.detach().cpu().numpy().squeeze()
                        joints_body_rec = joints_rec[:, :55, :]
                        joints_body_rec = joints_body_rec.reshape(-1, joints_body_rec.shape[1] * joints_body_rec.shape[2])

                        _ = l1_calculator.run(joints_body_rec.copy())

                        # Beat consistency calculation
                        if alignmenter is not None:
                            in_audio_eval = cond["y"]["in_audio_resample"][indx_samp].numpy()
                            a_offset = int(align_mask * (eval_dataloader.dataset.args.audio_sr / eval_dataloader.dataset.args.pose_fps))

                            onset_bt = alignmenter.load_audio(
                                in_audio_eval[: int(eval_dataloader.dataset.args.audio_sr / eval_dataloader.dataset.args.pose_fps * n_frames)],
                                a_offset,
                                len(in_audio_eval) - a_offset,
                                True,
                            )

                            beat_vel = alignmenter.load_pose(joints_body_rec, align_mask, n_frames - align_mask, 30, True)

                            align += alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n_frames - 2 * align_mask)

                        # Generate video comparison
                        if 0 and plot_video and index_batch < 3:  # Limit to first 3 batches to avoid too many videos
                            try:
                                # Get audio and text data for this sample
                                audio_data = None
                                text_data = None

                                if "in_audio_resample" in cond["y"]:
                                    audio_data = cond["y"]["in_audio_resample"][indx_samp].numpy()

                                if "tokens" in cond["y"]:
                                    tokens = cond["y"]["tokens"][indx_samp]
                                    if hasattr(eval_dataloader.dataset, '_create_text_from_in_word'):
                                        text_data = [eval_dataloader.dataset._create_text_from_in_word(tokens[:90])]
                                    else:
                                        text_data = ["Generated Motion"]

                                generate_video_comparison(
                                    gt_motion=motion[indx_samp],
                                    gt_poses=tar_pose[indx_samp, :n_frames],
                                    gt_trans=tar_trans[indx_samp, :n_frames],
                                    gt_betas=tar_beta[indx_samp, 0],
                                    gt_expressions=tar_exps[indx_samp, :n_frames],
                                    rec_motion=sample[indx_samp],
                                    rec_poses=rec_pose[indx_samp, :n_frames],
                                    rec_trans=rec_trans[indx_samp, :n_frames],
                                    rec_betas=tar_beta[indx_samp, 0],  # Use same betas
                                    rec_expressions=tar_exps[indx_samp, :n_frames],  # Use same expressions
                                    output_dir=out_path,
                                    person_name=person_name,
                                    batch_idx=index_batch,
                                    sample_idx=indx_samp,
                                    device=device,
                                    dataset_args=eval_dataloader.dataset.args,
                                    smplx_params=eval_dataloader.dataset.loaded_args["smplx_params"],
                                    plot_video=plot_video,
                                    audio_data=audio_data,
                                    text_data=text_data
                                )
                            except Exception as e:
                                print(f"⚠️ Failed to generate video for {person_name} batch {index_batch} sample {indx_samp}: {e}")

                    total_length += (n_frames - 2 * align_mask) * bs

            # Calculate final metrics for this person
            latent_out_all = np.concatenate(latent_out, axis=0)
            latent_ori_all = np.concatenate(latent_ori, axis=0)
            fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)

            # Store cross-person FID data
            cross_person_fid_data[person_name]["model_results"].append(latent_out_all)
            cross_person_fid_data[person_name]["gt_results"].append(latent_ori_all)

            jerk_vel_score = np.mean(jerk_vel)
            jerk_vel_diff_score = np.mean(jerk_vel_diff)
            avg_velocities_diff_score = np.mean(avg_velocities_diff)
            avg_accelerations_diff_score = np.mean(avg_accelerations_diff)
            maj_score = np.mean(maj_values)
            maj_diff_score = np.mean(maj_diffs)
            acc_cont_score = np.mean(acc_cont_values)
            acc_cont_diff_score = np.mean(acc_cont_diffs)
            l1div = l1_calculator.avg()

            if alignmenter is not None:
                align_avg = align / total_length
            else:
                align_avg = 0

            # Store results for this person
            all_results[person_name] = {
                "fid": fid,
                "l1div": l1div,
                "bc": align_avg,
                "jerk_vel": jerk_vel_score,
                "jerk_vel_diff": jerk_vel_diff_score,
                "avg_velocities_diff": avg_velocities_diff_score,
                "avg_accelerations_diff": avg_accelerations_diff_score,
                "maj": maj_score,
                "maj_diff": maj_diff_score,
                "acc_continuity": acc_cont_score,
                "acc_continuity_diff": acc_cont_diff_score,
                "checkpoint_path": checkpoint_path,
                "dataset_cache_path": dataset_cache_path,
            }

            print(f"✅ {person_name} evaluation completed:")
            print(f"   FID: {fid:.4f}")
            print(f"   L1Div: {l1div:.4f}")
            print(f"   BC: {align_avg:.4f}")
            print(f"   Jerk Vel: {jerk_vel_score:.4f}")
            print(f"   Jerk Vel Diff: {jerk_vel_diff_score:.4f}")
            print(f"   Avg Vel Diff: {avg_velocities_diff_score:.4f}")
            print(f"   Avg Acc Diff: {avg_accelerations_diff_score:.4f}")
            print(f"   MAJ: {maj_score:.4f}")
            print(f"   MAJ Diff: {maj_diff_score:.4f}")
            print(f"   AccC: {acc_cont_score:.4f}")
            print(f"   AccC Diff: {acc_cont_diff_score:.4f}")

        # except Exception as e:
        #     all_results[person_name] = {"error": str(e)}
        #     continue

    # Generate summary
    generate_multi_checkpoint_summary(all_results, cross_person_fid_data, out_path)

    print(f"\n🎉 Multi-checkpoint evaluation completed!")
    print(f"📁 Results saved to: {out_path}")


def generate_multi_checkpoint_summary(all_results, cross_person_fid_data, out_path):
    """Generate summary of multi-checkpoint evaluation results."""
    print(f"\n📊 MULTI-CHECKPOINT EVALUATION SUMMARY:")

    # Create summary table
    successful_results = {k: v for k, v in all_results.items() if "error" not in v}

    if not successful_results:
        print("❌ No successful evaluations found!")
        return

    # Calculate averages
    fids = [r["fid"] for r in successful_results.values()]
    l1divs = [r["l1div"] for r in successful_results.values()]
    bcs = [r["bc"] for r in successful_results.values()]
    jerk_vels = [r["jerk_vel"] for r in successful_results.values()]
    jerk_vel_diffs = [r["jerk_vel_diff"] for r in successful_results.values()]
    avg_velocities_diffs = [r["avg_velocities_diff"] for r in successful_results.values()]
    avg_accelerations_diffs = [r["avg_accelerations_diff"] for r in successful_results.values()]
    majs = [r["maj"] for r in successful_results.values()]
    maj_diffs = [r["maj_diff"] for r in successful_results.values()]
    acc_conts = [r["acc_continuity"] for r in successful_results.values()]
    acc_cont_diffs = [r["acc_continuity_diff"] for r in successful_results.values()]

    # Print summary table
    header = (
        f"{'Person':<12} {'FID':<10} {'L1Div':<10} {'BC':<10} {'Jerk Vel':<10} "
        f"{'Jerk Vel Diff':<15} {'Avg Vel Diff':<15} {'Avg Acc Diff':<15} {'MAJ':<10} "
        f"{'MAJ Diff':<12} {'AccC':<10} {'AccC Diff':<12}"
    )
    print(header)
    print("-" * len(header))

    for person_name, results in successful_results.items():
        print(
            f"{person_name:<12} "
            f"{results['fid']:<10.3f} {results['l1div']:<10.3f} {results['bc']:<10.3f} "
            f"{results['jerk_vel']:<10.3f} {results['jerk_vel_diff']:<15.3f} "
            f"{results['avg_velocities_diff']:<15.3f} {results['avg_accelerations_diff']:<15.3f} "
            f"{results['maj']:<10.3f} {results['maj_diff']:<12.3f} "
            f"{results['acc_continuity']:<10.3f} {results['acc_continuity_diff']:<12.3f}"
        )

    # Print averages
    print("-" * len(header))
    print(
        f"{'AVERAGE':<12} "
        f"{np.mean(fids):<10.3f} {np.mean(l1divs):<10.3f} {np.mean(bcs):<10.3f} "
        f"{np.mean(jerk_vels):<10.3f} {np.mean(jerk_vel_diffs):<15.3f} "
        f"{np.mean(avg_velocities_diffs):<15.3f} {np.mean(avg_accelerations_diffs):<15.3f} "
        f"{np.mean(majs):<10.3f} {np.mean(maj_diffs):<12.3f} "
        f"{np.mean(acc_conts):<10.3f} {np.mean(acc_cont_diffs):<12.3f}"
    )

    # Cross-person FID analysis
    if cross_person_fid_data:
        print(f"\n🔄 CROSS-PERSON FID ANALYSIS:")

        # Collect all model and GT results across persons
        all_model_results = []
        all_gt_results = []
        person_model_results = {}

        for person_name, person_data in cross_person_fid_data.items():
            if person_data["model_results"] and person_data["gt_results"]:
                all_model_results.extend(person_data["model_results"])
                all_gt_results.extend(person_data["gt_results"])
                person_model_results[person_name] = person_data["model_results"]

        if all_model_results and all_gt_results:
            # Calculate FID between model results and GT across all persons
            all_model_results_concat = np.concatenate(all_model_results, axis=0)
            all_gt_results_concat = np.concatenate(all_gt_results, axis=0)

            cross_person_fid = data_tools.FIDCalculator.frechet_distance(all_model_results_concat, all_gt_results_concat)
            print(f"   Cross-Person FID (Model vs GT): {cross_person_fid:.4f}")

            # Calculate FID between GT and RES for each person
            person_names = list(person_model_results.keys())
            cross_person_fids = np.zeros((len(person_names), len(person_names)))

            for i, person1 in enumerate(person_names):
                for j, person2 in enumerate(person_names):
                    # Compare GT of person1 with RES of person2
                    person1_gt_results = np.concatenate(cross_person_fid_data[person1]["gt_results"], axis=0)
                    person2_res_results = np.concatenate(person_model_results[person2], axis=0)
                    cross_person_fids[i, j] = data_tools.FIDCalculator.frechet_distance(person1_gt_results, person2_res_results)

            # Print cross-person FID table
            print(f"\n   Cross-Person FID Matrix:")
            fid_table = pd.DataFrame(cross_person_fids, index=[f"{p} GT" for p in person_names], columns=[f"{p} RES" for p in person_names])
            print(fid_table)
            fid_table.to_csv(os.path.join(out_path, "cross_person_fid_table.csv"))

    # Save successful results as CSV
    csv_path = os.path.join(out_path, "successful_results.csv")
    df = pd.DataFrame.from_dict(successful_results, orient='index')
    df.index.name = 'person_name'
    df.to_csv(csv_path)
    print(f"✅ Successful results table saved to: {csv_path}")

    # Save results to JSON
    summary_data = {
        "summary": {
            "total_persons": len(all_results),
            "successful_persons": len(successful_results),
            "failed_persons": len(all_results) - len(successful_results),
            "avg_fid": float(np.mean(fids)),
            "avg_l1div": float(np.mean(l1divs)),
            "avg_bc": float(np.mean(bcs)),
            "avg_jerk_vel": float(np.mean(jerk_vels)),
            "avg_jerk_vel_diff": float(np.mean(jerk_vel_diffs)),
            "avg_velocities_diff": float(np.mean(avg_velocities_diffs)),
            "avg_accelerations_diff": float(np.mean(avg_accelerations_diffs)),
            "avg_maj": float(np.mean(majs)),
            "avg_maj_diff": float(np.mean(maj_diffs)),
            "avg_acc_continuity": float(np.mean(acc_conts)),
            "avg_acc_continuity_diff": float(np.mean(acc_cont_diffs)),
        },
        "detailed_results": all_results,
        "cross_person_fid": cross_person_fid_data if cross_person_fid_data else None,
    }

    summary_path = os.path.join(out_path, "multi_checkpoint_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n📁 Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Beat2 Evaluation Script")
    parser.add_argument(
        "--validated_mapping_json", type=str, help="JSON file containing validated mapping of person names to (checkpoint_path, dataset_cache_path)"
    )
    parser.add_argument(
        "--plot_video_evaluation", action="store_true", help="Generate comparison videos during evaluation"
    )
    parser.add_argument(
        "--load_generated_motion", type=str, required=False, help="Load generated motion from NPZ file instead of sampling from model"
    )

    # Get arguments
    args = evaluate_args()
    emage_args = config.parse_args()

    # Parse custom arguments
    custom_args, unknown = parser.parse_known_args()

    # Fix seed
    fixseed(args.seed)

    load_generated_motion = getattr(custom_args, "load_generated_motion", False)
    # Check if multi-checkpoint evaluation is requested
    if custom_args.validated_mapping_json:
        print(f"🔄 Multi-checkpoint evaluation mode")
        plot_video = getattr(custom_args, "plot_video_evaluation", False)
        run_multi_checkpoint_evaluation(args, custom_args.validated_mapping_json, plot_video=plot_video, load_generated_motion=load_generated_motion)
        exit(0)

    # Set fps and number of frames to generate
    fps: float = 30
    max_frames: int = 196
    n_frames: int = min(max_frames, int(args.motion_length * fps))

    # Set Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  # Replace "0" with the GPU id you want to be visible
    device: int = args.device

    # args.models_names = "model000038000.pt,model000001000.pt"
    if args.models_folder is None:
        models_paths = [args.model_path]
        # Create folder for evaluation model
        niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
        evaluation_folder: str = os.path.join(os.path.dirname(args.model_path), f"evaluation_{niter}")
        os.makedirs(evaluation_folder, exist_ok=True)
    else:
        if args.models_names != "":
            models_paths = [os.path.join(args.models_folder, model_name) for model_name in args.models_names.split(",")]
            evaluation_folder: str = os.path.join(args.models_folder, f"evaluation_{args.models_names}")
        else:
            path_pattern = os.path.join(args.models_folder, "model*.pt")
            # Use glob to find all files matching the pattern
            models_paths = glob.glob(path_pattern)
            evaluation_folder: str = os.path.join(args.models_folder, f"evaluation_all_models-in_folder")
        os.makedirs(evaluation_folder, exist_ok=True)
        models_paths.sort()

    # Create folder for specific timestamp
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    out_path: str = os.path.join(evaluation_folder, f"{timestamp}_seed_{args.seed}")
    os.makedirs(out_path, exist_ok=True)

    logger.configure(dir=out_path)

    fids = []
    l1divs = []
    bcs = []
    jerk_vels = []
    jerk_vel_diffs = []
    avg_velocities_diffs = []
    avg_accelerations_diffs = []
    maj_scores = []
    maj_diff_scores = []
    acc_cont_scores = []
    acc_cont_diff_scores = []
    models_names = []
    logger.info(f"len models paths: {len(models_paths)}")
    batch_size = 32
    for indx_model_path, model_path in enumerate(models_paths):
        logger.info("Loading dataset...")
        eval_dataloader = get_dataset_loader(
            name=args.dataset,
            batch_size=batch_size,
            split="val",
            device=device,
        )

        logger.info("Creating model and diffusion...")
        model, diffusion = load_model(args, dist_util.dev(), model_path=model_path)

        model_name = os.path.basename(model_path).strip(".pt")
        n_joints: int = 55
        align_mask = 60
        l1_calculator = metric.L1div()
        align = 0
        total_length = 0
        latent_out = []
        latent_ori = []
        jerk_vel = []
        jerk_vel_diff = []
        avg_velocities_diff = []
        avg_accelerations_diff = []
        maj_batch_values = []
        maj_batch_diffs = []
        acc_cont_batch_values = []
        acc_cont_batch_diffs = []
        eval_model = load_evaluation_model(emage_args, device)
        eval_model.to(device)
        eval_model.eval()

        # alignmenter = None  # metric.alignment(0.3, 7, eval_dataloader.dataset.avg_vel)  # only on uper body  # todo - change to train
        alignmenter = metric.alignment(0.3, 7, eval_dataloader.dataset.avg_vel, upper_body=[3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        with torch.no_grad():
            for index_batch, (motion, cond) in tqdm(enumerate(eval_dataloader)):
                print(f"index_batch {index_batch} / {len(eval_dataloader)}")

                motion = motion.to(device)

                bs, n, j = motion.shape[0], motion.shape[-1], n_joints

                cond["y"] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in cond["y"].items()}
                tar_exps = torch.stack(cond["y"]["tar_exps"]).to(device)
                tar_beta = torch.stack(cond["y"]["tar_beta"]).to(device)
                tar_pose = torch.stack(cond["y"]["tar_pose"]).to(device)
                tar_trans = torch.stack(cond["y"]["tar_trans"]).to(device)

                avg_velocities_gt, avg_accelerations_gt, jerk_vel_gt_motions = calculate_speed_3d(tar_trans.cpu().numpy(), 1 / fps)
                maj_gt = calculate_mean_absolute_jerk(tar_trans.cpu().numpy(), 1 / fps)
                acc_cont_gt = calculate_acceleration_continuity(tar_trans.cpu().numpy(), 1 / fps)
                tar_pose_mat = rc.axis_angle_to_matrix(tar_pose.reshape(bs * n, j, 3))
                tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_mat).reshape(bs, n, j * 6)

                use_test_data = False
                if use_test_data:
                    # split to chunks
                    pre_frames = args.handshake_size  # data.dataset.args.pre_frames
                    roundt = (n - pre_frames) // (eval_dataloader.dataset.args.pose_length - pre_frames)
                    remain = (n - pre_frames) % (eval_dataloader.dataset.args.pose_length - pre_frames)
                    round_l = eval_dataloader.dataset.args.pose_length - pre_frames

                    audio_fps = eval_dataloader.dataset.args.audio_fps
                    pose_fps = eval_dataloader.dataset.args.pose_fps
                    audio_data = cond["y"]["in_audio_resample"][0]
                    round_model_kwargs = {
                        "y": {"tokens": [], "text": [], "audio": [], "lengths": torch.zeros(roundt, device=dist_util.dev()).int()}
                    }
                    tokens_orig = cond["y"]["tokens"][0]
                    audio_orig = cond["y"]["audio"][0]
                    for i in range(0, roundt):
                        in_word_tmp = tokens_orig[i * (round_l) : (i + 1) * (round_l) + pre_frames]
                        round_model_kwargs["y"]["tokens"].append(in_word_tmp)
                        round_model_kwargs["y"]["text"].append(eval_dataloader.dataset._create_text_from_in_word(in_word_tmp))
                        in_audio_tmp = audio_orig[
                            i * (audio_fps // pose_fps * round_l) : (i + 1) * (audio_fps // pose_fps * round_l)
                            + audio_fps // pose_fps * pre_frames
                        ]
                        round_model_kwargs["y"]["audio"].append(in_audio_tmp)
                        # if i == roundt - 1:
                        #     round_model_kwargs["y"]["lengths"][i] = remain
                        round_model_kwargs["y"]["lengths"][i] = eval_dataloader.dataset.args.pose_length

                remain = n % emage_args.vae_test_len

                latent_ori.append(eval_model.map2latent(tar_pose_6d[:, : n - remain]).reshape(-1, emage_args.vae_length).cpu().numpy())
                # continue
                # latent_ori.append(tar_pose.cpu().numpy())

                # Load generated motion from NPZ file or sample from model
                if load_generated_motion:
                    print(f"📁 Loading generated motion from: {load_generated_motion}")
                    try:
                        # Load pre-generated motion from NPZ file
                        npz_data = np.load(load_generated_motion)
                        sample = torch.from_numpy(npz_data["motion"]).to(device)
                        print(f"✅ Loaded motion shape: {sample.shape}")
                    except Exception as e:
                        print(f"❌ Failed to load NPZ file: {e}")
                        print("🔄 Falling back to model sampling...")
                        sample_fn = diffusion.p_sample_loop
                        sample = sample_fn(
                            model,
                            (
                                batch_size,
                                model.njoints,
                                model.nfeats,
                                max_frames,
                            ),
                            clip_denoised=False,
                            model_kwargs=cond,
                            skip_timesteps=0,
                            init_image=None,
                            progress=True,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                        )
                else:
                    # Sample from model
                    sample_fn = diffusion.p_sample_loop
                    sample = sample_fn(
                        model,
                        (
                            batch_size,
                            model.njoints,
                            model.nfeats,
                            max_frames,
                        ),
                        clip_denoised=False,
                        model_kwargs=cond,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=True,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                    )

                # Inverse normalization
                sample = sample.squeeze(dim=2).permute(0, 2, 1)  # [337, n_frames] -> [n_frames, 337]
                # sample.reshape(bs, n, -1).shape
                sample = eval_dataloader.dataset.inv_transform(sample)

                # Split the network output to the different parts
                rotations_6d = sample[:, :, : n_joints * 6]
                translations = sample[:, :, n_joints * 6 : n_joints * 6 + 3]
                # contact = sample[:, -4:]

                # Convert rot6d to pose 3d
                n_frames: int = rotations_6d.shape[1]
                rotations_matrix = rc.rotation_6d_to_matrix(rotations_6d.reshape(bs, n_frames, n_joints, 6))
                rotations_angle = rc.matrix_to_axis_angle(rotations_matrix).reshape(bs, n_frames, n_joints * 3)

                rec_pose = rotations_angle.to(device)
                rec_trans = translations.to(device)
                avg_velocities_rec, avg_accelerations_rec, jerk_vel_motions = calculate_speed_3d(translations.cpu().numpy(), 1 / fps)
                maj_rec = calculate_mean_absolute_jerk(translations.cpu().numpy(), 1 / fps)
                acc_cont_rec = calculate_acceleration_continuity(translations.cpu().numpy(), 1 / fps)
                jerk_vel.append(jerk_vel_motions)
                avg_velocities_diff.append(np.abs(avg_velocities_gt - avg_velocities_rec))
                avg_accelerations_diff.append(np.abs(avg_accelerations_gt - avg_accelerations_rec))
                jerk_vel_diff.append(np.abs(jerk_vel_gt_motions - jerk_vel_motions))
                maj_batch_values.append(maj_rec)
                maj_batch_diffs.append(np.abs(maj_gt - maj_rec))
                acc_cont_batch_values.append(acc_cont_rec)
                acc_cont_batch_diffs.append(np.abs(acc_cont_gt - acc_cont_rec))
                latent_out.append(
                    eval_model.map2latent(rotations_6d.reshape(bs, n, -1)[:, : n - remain].to(device))
                    .reshape(-1, emage_args.vae_length)
                    .cpu()
                    .numpy()
                )

                for indx_samp in range(bs):
                    # L1 Diversity is focused solely on local motion that is why transl is 0.
                    smplx_output_rec = eval_dataloader.dataset.smplx(
                        betas=tar_beta[indx_samp, :n_frames, :],
                        transl=rec_trans[indx_samp] - rec_trans[indx_samp],
                        expression=tar_exps[indx_samp, :n_frames, :] - tar_exps[indx_samp, :n_frames, :],
                        jaw_pose=rec_pose[indx_samp, :, 66:69],
                        global_orient=rec_pose[indx_samp, :, :3],
                        body_pose=rec_pose[indx_samp, :, 3 : 21 * 3 + 3],
                        left_hand_pose=rec_pose[indx_samp, :, 25 * 3 : 40 * 3],
                        right_hand_pose=rec_pose[indx_samp, :, 40 * 3 : 55 * 3],
                        return_joints=True,
                        leye_pose=rec_pose[indx_samp, :, 69:72],
                        reye_pose=rec_pose[indx_samp, :, 72:75],
                        return_verts=True,
                    )

                    vertices_rec = smplx_output_rec.vertices.detach().cpu().numpy().squeeze()
                    joints_rec = smplx_output_rec.joints.detach().cpu().numpy().squeeze()
                    joints_body_rec = joints_rec[:, :55, :]
                    joints_body_rec = joints_body_rec.reshape(-1, joints_body_rec.shape[1] * joints_body_rec.shape[2])

                    _ = l1_calculator.run(joints_body_rec.copy())

                    if alignmenter is not None:
                        in_audio_eval = cond["y"]["in_audio_resample"][indx_samp].numpy()
                        a_offset = int(align_mask * (eval_dataloader.dataset.args.audio_sr / eval_dataloader.dataset.args.pose_fps))

                        onset_bt = alignmenter.load_audio(
                            in_audio_eval[: int(eval_dataloader.dataset.args.audio_sr / eval_dataloader.dataset.args.pose_fps * n_frames)],
                            a_offset,
                            len(in_audio_eval) - a_offset,
                            True,
                        )

                        beat_vel = alignmenter.load_pose(joints_body_rec, align_mask, n_frames - align_mask, 30, True)

                        align += alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n_frames - 2 * align_mask)
                total_length += (n_frames - 2 * align_mask) * bs

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        fids.append(fid)
        jerk_vel_score = np.mean(jerk_vel)
        jerk_vels.append(jerk_vel_score)
        jerk_vel_diff_score = np.mean(jerk_vel_diff)
        jerk_vel_diffs.append(jerk_vel_diff_score)
        avg_velocities_diff_score = np.mean(avg_velocities_diff)
        avg_velocities_diffs.append(avg_velocities_diff_score)
        avg_accelerations_diff_score = np.mean(avg_accelerations_diff)
        avg_accelerations_diffs.append(avg_accelerations_diff_score)
        maj_score = np.mean(maj_batch_values)
        maj_scores.append(maj_score)
        maj_diff_score = np.mean(maj_batch_diffs)
        maj_diff_scores.append(maj_diff_score)
        acc_cont_score = np.mean(acc_cont_batch_values)
        acc_cont_scores.append(acc_cont_score)
        acc_cont_diff_score = np.mean(acc_cont_batch_diffs)
        acc_cont_diff_scores.append(acc_cont_diff_score)
        l1div = l1_calculator.avg()
        l1divs.append(l1div)
        if alignmenter is not None:
            align_avg = align / total_length
        else:
            align_avg = 0
        bcs.append(align_avg)

        models_names.append(model_name)

        logger.info(f"align score: {align_avg}")
        logger.info(f"evaluation for model: {model_name}")
        logger.info(f"fid score (smaller better): {fid}")
        logger.info(f"jerk_vels score: {jerk_vel_score}")
        logger.info(f"avg_velocities_diff score: {avg_velocities_diff_score}")
        logger.info(f"avg_accelerations_diff score: {avg_accelerations_diff_score}")
        logger.info(f"diff jerk_vels score (smaller better): {jerk_vel_diff_score}")
        logger.info(f"MAJ score: {maj_score}")
        logger.info(f"MAJ diff score: {maj_diff_score}")
        logger.info(f"AccC score: {acc_cont_score}")
        logger.info(f"AccC diff score: {acc_cont_diff_score}")
        logger.info(f"l1div score (heigher better): {l1div}")

    # Create a header
    header = (
        f"{'Model Name':<10} {'FID':<10} {'L1Div':<10} {'BC':<10} {'Jerk Vel':<10} "
        f"{'Jerk Vel Diff':<15} {'Average Vel Diff':<17} {'Average acc Diff':<17} "
        f"{'MAJ':<10} {'MAJ Diff':<12} {'AccC':<10} {'AccC Diff':<12}"
    )

    # Log the header
    logger.info(header)

    # Combine the lists into rows and format them
    for row in zip(
        models_names,
        fids,
        l1divs,
        bcs,
        jerk_vels,
        jerk_vel_diffs,
        avg_velocities_diffs,
        avg_accelerations_diffs,
        maj_scores,
        maj_diff_scores,
        acc_cont_scores,
        acc_cont_diff_scores,
    ):
        logger.info(
            f"{row[0]:<10} {row[1]:<10.3f} {row[2]:<10.3f} {row[3]:<10.3f} {row[4]:<10.3f} "
            f"{row[5]:<15.3f} {row[6]:<15.3f} {row[7]:<15.3f} {row[8]:<10.3f} {row[9]:<12.3f} {row[10]:<10.3f} {row[11]:<12.3f}"
        )

    # Creating the graph
    plt.figure(figsize=(10, 5))

    # Plotting FID Score
    plt.plot(models_names, fids, marker="o", linestyle="-", color="b", label="FID Score (v)")
    # Plotting align Score
    plt.plot(models_names, bcs, marker="^", linestyle="-", color="g", label="align Score (^)")
    # # Plotting jerk_vels Score
    # plt.plot(
    #     models_names,
    #     jerk_vels,
    #     marker="o",
    #     linestyle="-",
    #     color="y",
    #     label="jerk velocity Score",
    # )
    # # Plotting jerk_vel_diffs Score
    # plt.plot(
    #     models_names,
    #     jerk_vel_diffs,
    #     marker="x",
    #     linestyle="--",
    #     color="r",
    #     label="jerk velocity diffrence Score (v)",
    # )

    # Plotting avg_velocities_diffs Score
    plt.plot(
        models_names,
        avg_velocities_diffs,
        marker="x",
        linestyle="--",
        color="orange",
        label="avg velocities diffrence Score (v)",
    )

    # Plotting avg_accelerations_diffs Score
    plt.plot(
        models_names,
        avg_accelerations_diffs,
        marker="x",
        linestyle="--",
        color="red",
        label="avg accelerations diffrence Score (v)",
    )

    # Plotting L1 Diversity Score
    plt.plot(
        models_names,
        l1divs,
        marker="s",
        linestyle="-",
        color="purple",
        label="L1 Diversity Score (^)",
    )

    # Adding title, labels, and grid
    plt.title("Model Performance Improvement Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)
    plt.xticks(rotation=15)
    plt.legend()

    # Save the plot to a file
    plt.savefig(os.path.join(out_path, f"Model_Evaluation{str(datetime.now())}.png"))

    # Show the plot
    plt.show()

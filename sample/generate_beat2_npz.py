# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.get_data import get_dataset_loader
from data_loaders.beat2.utils.build_vocab import Vocab
from data_loaders.beat2.utils import rotation_conversions as rc
import datetime
import numpy as np


def main_create_npz():

    # Get arguments
    args = generate_args()

    # Fix seed
    fixseed(args.seed)

    # Set number of frames to generate
    max_frames: int = 196

    # Set Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  # Replace "0" with the GPU id you want to be visible
    device: int = args.device
    batch_size = args.batch_size

    # Create folder for specific model
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    results_folder: str = os.path.join(os.path.dirname(args.model_path), f"results_{niter}")
    os.makedirs(results_folder, exist_ok=True)

    # Create folder for specific timestamp
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    out_path: str = os.path.join(results_folder, f"{timestamp}_seed_{args.seed}_npz")
    os.makedirs(out_path, exist_ok=True)

    print("Loading dataset...")
    test_data_loader = get_dataset_loader(name=args.dataset, batch_size=batch_size, split="test", device=device)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, test_data_loader)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    model.to(device)

    n_joints: int = 55

    def sample_to_pose_trans(sample):
        # Inverse normalization
        sample = sample.squeeze().permute(0, 2, 1)  # [337, n_frames] -> [n_frames, 337]
        sample = test_data_loader.dataset.inv_transform(sample)

        # Split the network output to the different parts
        rotations_6d = sample[:, :, : n_joints * 6]
        translations = sample[:, :, n_joints * 6 : n_joints * 6 + 3]
        # contact = sample[:, -4:]

        # Convert rot6d to pose 3d
        n_frames: int = rotations_6d.shape[1]
        rotations_matrix = rc.rotation_6d_to_matrix(rotations_6d.reshape(batch_size, n_frames, n_joints, 6))
        rotations_angle = rc.matrix_to_axis_angle(rotations_matrix).reshape(batch_size, n_frames, n_joints * 3)

        pose = rotations_angle.to(device)
        trans = translations.to(device)

        return pose, trans, n_frames

    # iterator = iter(test_data_loader)
    # motion, model_kwargs = next(iterator)

    with torch.no_grad():
        for index_batch, (motion, model_kwargs) in enumerate(test_data_loader):
            names = np.squeeze(model_kwargs["y"]["tar_name"])
            n_frames = motion.shape[-1]
            clip_frames = min(max_frames, n_frames)

            # Get GT parammeters from data
            tar_exps = torch.stack(model_kwargs["y"]["tar_exps"])[:, :clip_frames, :].to(device)
            tar_beta = torch.stack(model_kwargs["y"]["tar_beta"])[:, :clip_frames, :].to(device)
            tar_pose = torch.stack(model_kwargs["y"]["tar_pose"])[:, :clip_frames, :].to(device)
            tar_trans = torch.stack(model_kwargs["y"]["tar_trans"])[:, :clip_frames, :].to(device)
            # tar_trans = torch.zeros_like(tar_trans)

            # Create GT plot
            # Get position and transition from data
            sample = motion.to(device)

            # Create model prediction plot
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
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Get position and transition from model
            rec_pose, rec_trans, n_frames = sample_to_pose_trans(sample)

            # Clip the betas and expr from the start of the audio - #TODO: select the betas/expr in a more smart way
            clip_frames = min(max_frames, n_frames)
            rec_pose = rec_pose[:, :clip_frames, :]
            rec_trans = rec_trans[:, :clip_frames, :]
            # rec_trans = torch.zeros_like(rec_trans[:clip_frames, :])

            # create npz (for blender)
            tar_pose_np = tar_pose.detach().cpu().numpy()
            rec_pose_np = rec_pose.detach().cpu().numpy()
            rec_trans_np = rec_trans.detach().cpu().numpy().reshape(batch_size * n_frames, 3)
            rec_exp_np = tar_exps.detach().cpu().numpy().reshape(batch_size * n_frames, 100)
            tar_exp_np = tar_exps.detach().cpu().numpy().reshape(batch_size * n_frames, 100)
            tar_trans_np = tar_trans.detach().cpu().numpy().reshape(batch_size * n_frames, 3)
            tar_beta_np = tar_beta[0].detach().cpu().numpy()

            year_smplx = test_data_loader.dataset.smplx.gender.split("_")[-1]
            for i in range(batch_size):
                name = names[i]

                np.savez(
                    out_path + "/gt_" + name + ".npz",
                    betas=tar_beta_np,
                    # betas=gt_npz["betas"],
                    poses=tar_pose_np[i],
                    expressions=tar_exp_np[i],
                    trans=tar_trans_np[i],
                    model=f"smplx{year_smplx}",
                    gender=test_data_loader.dataset.smplx.gender,
                    mocap_frame_rate=test_data_loader.dataset.args.render_video_fps,
                )
                np.savez(
                    out_path + "/res_" + name + ".npz",
                    betas=tar_beta_np[i],
                    # betas=gt_npz["betas"],
                    poses=rec_pose_np[i],
                    expressions=rec_exp_np[i],
                    trans=rec_trans_np[i],
                    model=f"smplx{year_smplx}",
                    gender=test_data_loader.dataset.smplx.gender,
                    mocap_frame_rate=test_data_loader.dataset.args.render_video_fps,
                )


if __name__ == "__main__":
    main_create_npz()

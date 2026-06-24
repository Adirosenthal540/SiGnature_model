# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import datetime
import numpy as np
import torch
from utils.fixseed import fixseed
from utils.model_util import load_model
from utils import dist_util
from utils.parser_util import generate_args
from data_loaders.get_data import get_dataset_loader
from data_loaders.beat2.utils import rotation_conversions as rc
# from data_loaders.beat2.utils.other_tools_hf import render_one_sequence
from data_loaders.beat2.utils.media import add_subtitles_to_video, add_audio_to_video, create_srt_file
from data_loaders.beat2.utils.cache_utils import calculate_mean_std
from data_loaders.beat2.utils.build_vocab import Vocab

os.environ["PYOPENGL_PLATFORM"] = "egl"


def main():

    # Get arguments
    args = generate_args()

    # Fix seed
    fixseed(args.seed)

    # Set number of frames to generate
    chank_frames: int = 196

    # Set Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  # Replace "0" with the GPU id you want to be visible
    device: int = args.device
    args.batch_size = 1
    batch_size: int = args.batch_size
    # Create folder for specific model
    model_path = args.model_path
    niter = os.path.basename(model_path).replace("model", "").replace(".pt", "")
    results_folder: str = os.path.join(os.path.dirname(model_path), f"results_{niter}")
    os.makedirs(results_folder, exist_ok=True)

    # Create folder for specific timestamp
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    out_path: str = os.path.join(results_folder, f"{timestamp}_seed_{args.seed}")
    os.makedirs(out_path, exist_ok=True)

    # calculate_mean_std()

    print("Loading dataset...")
    test_data_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, split="test", device=args.device, shuffle=False)
    print("Creating model and diffusion...")
    model, diffusion = load_model(args, device)

    # Get person parammeters from data
    iterator = iter(test_data_loader)
    motion, model_kwargs = next(iterator)

    name = test_data_loader.dataset.selected_files[0]

    tar_exps = model_kwargs["y"]["tar_exps"][0].to(device)
    tar_beta = model_kwargs["y"]["tar_beta"][0].to(device)
    tokens_orig = model_kwargs["y"]["tokens"][0]

    # Get audio text
    text: str = model_kwargs["y"]["text"][0]

    # Get audio
    audio_data = model_kwargs["y"]["in_audio_resample"][0]
    audio_sr = test_data_loader.dataset.args.audio_sr

    # Animation save text
    animation_save_path = os.path.join(out_path, f"video.mp4")

    def sample_to_pose_trans(sample):
        # Inverse normalization
        sample = sample.squeeze().T  # [337, n_frames] -> [n_frames, 337]
        sample = test_data_loader.dataset.inv_transform(sample)

        # Split the network output to the different parts
        n_joints: int = 55
        rotations_6d = sample[:, : n_joints * 6]
        translations = sample[:, n_joints * 6 : n_joints * 6 + 3]
        contact = sample[:, -4:]

        # Convert rot6d to pose 3d
        n_frames: int = rotations_6d.shape[0]
        rotations_matrix = rc.rotation_6d_to_matrix(rotations_6d.reshape(n_frames, n_joints, 6))
        rotations_angle = rc.matrix_to_axis_angle(rotations_matrix).reshape(n_frames, n_joints * 3)

        pose = rotations_angle.to(device)
        trans = translations.to(device)

        return pose, trans, n_frames

    # Get position and transition from data
    sample = motion[0].to(device)

    tar_pose, tar_trans, n_frames = sample_to_pose_trans(sample)
    # clip_frames = min(196, n_frames)

    n = tar_pose.shape[0]
    pre_frames = test_data_loader.dataset.args.pre_frames
    roundt = (n - pre_frames) // (test_data_loader.dataset.args.pose_length - pre_frames)
    remain = (n - pre_frames) % (test_data_loader.dataset.args.pose_length - pre_frames)
    round_l = test_data_loader.dataset.args.pose_length - pre_frames

    sample_out = torch.zeros(
        (
            batch_size,
            model.njoints,
            model.nfeats,
            n,
        ),
        device=device,
    )

    for i in range(0, roundt):
        round_model_kwargs = {"y": {}}
        in_word_tmp = model_kwargs["y"]["tokens"][0][i * (round_l) : (i + 1) * (round_l) + pre_frames]
        round_model_kwargs["y"]["tokens"] = [in_word_tmp]
        round_model_kwargs["y"]["text"] = [test_data_loader.dataset._create_text_from_in_word(in_word_tmp)]
        in_audio_tmp = model_kwargs["y"]["audio"][0][i * (16000 // 30 * round_l) : (i + 1) * (16000 // 30 * round_l) + 16000 // 30 * pre_frames]
        round_model_kwargs["y"]["audio"] = [in_audio_tmp]
        round_model_kwargs["y"]["lengths"] = [chank_frames]
        round_model_kwargs["y"]["scale"] = torch.ones(1, device=dist_util.dev()) * args.guidance_param
        with torch.no_grad():
            # diffusion.eval()
            # Create model prediction plot
            sample_fn = diffusion.p_sample_loop
            sample = sample_fn(
                model,
                (
                    batch_size,
                    model.njoints,
                    model.nfeats,
                    test_data_loader.dataset.args.pose_length,
                ),  # BUG FIX - currently support only batch_size == 1
                clip_denoised=False,
                model_kwargs=round_model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            sample_out[:, :, :, i * (round_l) : (i + 1) * (round_l) + pre_frames] = sample

        # Get position and transition from model
    rec_pose, rec_trans, n_frames = sample_to_pose_trans(sample_out)

    # Create npz (for blender)
    tar_pose_np = tar_pose.detach().cpu().numpy()
    rec_pose_np = rec_pose.detach().cpu().numpy()
    rec_trans_np = rec_trans.detach().cpu().numpy().reshape(batch_size * n_frames, 3)
    rec_exp_np = tar_exps.detach().cpu().numpy().reshape(batch_size * n_frames, 100)
    tar_exp_np = tar_exps.detach().cpu().numpy().reshape(batch_size * n_frames, 100)
    tar_trans_np = tar_trans.detach().cpu().numpy().reshape(batch_size * n_frames, 3)
    tar_beta_np = tar_beta[0].detach().cpu().numpy()

    year_smplx = test_data_loader.dataset.loaded_args["smplx_params"]["gender"].split("_")[-1]

    gt_path: str = out_path + "/gt_" + name + ".npz"
    np.savez(
        gt_path,
        betas=tar_beta_np,
        # betas=gt_npz["betas"],
        poses=tar_pose_np,
        expressions=tar_exp_np,
        trans=tar_trans_np,
        model=f"smplx{year_smplx}",
        gender=test_data_loader.dataset.loaded_args["smplx_params"]["gender"],
        mocap_frame_rate=test_data_loader.dataset.loaded_args["render_video_fps"],
    )

    res_path: str = out_path + "/res_" + name + ".npz"
    np.savez(
        res_path,
        betas=tar_beta_np,
        # betas=gt_npz["betas"],
        poses=rec_pose_np,
        expressions=rec_exp_np,
        trans=rec_trans_np,
        model=f"smplx{year_smplx}",
        gender=test_data_loader.dataset.loaded_args["smplx_params"]["gender"],  # check its "neutral"
        mocap_frame_rate=test_data_loader.dataset.loaded_args["render_video_fps"],
    )

    # Create silient video
    silent_video_file_path: str = render_one_sequence(
        res_npz_path=res_path,
        gt_npz_path=gt_path,
        output_dir=out_path,
        smplx_params=test_data_loader.dataset.loaded_args["smplx_params"],
        args=test_data_loader.dataset.args,
        device=device,
    )

    # # Add Subtitle to video
    # base_filename_without_ext = os.path.splitext(os.path.basename(res_path))[0]
    # subtitled_video_file_path = os.path.join(out_path, f"{base_filename_without_ext}_with_sub.mp4")
    # add_subtitles(silent_video_file_path, subtitled_video_file_path, [text])
    # os.remove(silent_video_file_path)
    # Add Subtitle to video
    subtitles = []
    last_word = -1
    for i in range(0, n, test_data_loader.dataset.args.pose_length):
        start_index = i
        while tokens_orig[start_index] == last_word:
            start_index += 1
        in_word_tmp = tokens_orig[start_index : i + test_data_loader.dataset.args.pose_length]
        subtitles.append(test_data_loader.dataset._create_text_from_in_word(in_word_tmp))
        last_word = in_word_tmp[-1]
    base_filename_without_ext = os.path.splitext(os.path.basename(res_path))[0]
    subtitled_video_file_path = os.path.join(out_path, f"{base_filename_without_ext}_with_sub.mp4")
    add_subtitles(silent_video_file_path, subtitled_video_file_path, subtitles)
    os.remove(silent_video_file_path)

    # Add audio to video
    pose_fps = test_data_loader.dataset.args.pose_fps
    audio_data = audio_data[: int(np.floor(n_frames / pose_fps * audio_sr))]
    base_filename_without_ext = os.path.splitext(os.path.basename(res_path))[0]
    audio_video_file_path = os.path.join(out_path, f"{base_filename_without_ext}.mp4")
    add_audio_to_video(subtitled_video_file_path, audio_video_file_path, audio_data, audio_sr)
    os.remove(subtitled_video_file_path)


# def add_subtitles(audio_video_file_path: str, subtitled_video_file_path: str, subtitles: list):
#     output_srt_path = subtitled_video_file_path.replace(".mp4", ".srt")
#     create_srt_file(subtitles, output_srt_path)
#     add_subtitles_to_video(audio_video_file_path, output_srt_path, subtitled_video_file_path)


if __name__ == "__main__":
    main()

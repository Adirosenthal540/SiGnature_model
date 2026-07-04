from argparse import ArgumentParser
import argparse
import glob
import os
import json

OFFICIAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckp", "official_model")


def _resolve_person_model(person):
    """Resolve a --person identifier to the model .pt path in ckp/official_model/."""
    if not os.path.isdir(OFFICIAL_MODEL_DIR):
        raise ValueError(f"Official model directory not found: {OFFICIAL_MODEL_DIR}")

    matches = [d for d in os.listdir(OFFICIAL_MODEL_DIR)
               if d == person or d.split("_")[0] == person]
    if not matches:
        available = sorted(os.listdir(OFFICIAL_MODEL_DIR))
        raise ValueError(f"No model folder for person '{person}'. Available: {available}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous person '{person}', matches: {sorted(matches)}. Use the full name (e.g. '1_wayne').")

    folder = os.path.join(OFFICIAL_MODEL_DIR, matches[0])
    pt_files = sorted(glob.glob(os.path.join(folder, "model*.pt")))
    if not pt_files:
        raise ValueError(f"No model*.pt files found in {folder}")
    return pt_files[-1]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        val = v.lower()
        if val in ("yes", "true", "t", "y", "1"):
            return True
        if val in ("no", "false", "f", "n", "0"):
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_and_load_from_model(parser, model_path=None):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args, unknown = parser.parse_known_args()
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    args_to_overwrite.remove("diffusion_steps")
    args_to_overwrite.append("data_path")
    model_path = model_path if model_path is not None else args.model_path
    if model_path is not None:
        # load args from model
        if model_path is not None or args.models_folder is not None:
            if model_path is not None:
                args_path = os.path.join(os.path.dirname(model_path), "args.json")
            else:
                args_path = os.path.join(args.models_folder, "args.json")
        else:
            model_path = get_model_path_from_args()
            args_path = os.path.join(os.path.dirname(model_path), "args.json")
        assert os.path.exists(args_path), f"Arguments json file was not found! {args_path}"
        with open(args_path, "r") as fr:
            model_args = json.load(fr)

        for a in args_to_overwrite:
            if a in model_args.keys():
                setattr(args, a, model_args[a])

            elif "cond_mode" in model_args:  # backward compitability
                unconstrained = model_args["cond_mode"] == "no_cond"
                setattr(args, "unconstrained", unconstrained)

            else:
                print("Warning: was not able to load [{}], using default value [{}] instead.".format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError("group_name was not found.")


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError("model_path argument must be specified.")


def add_base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=12, type=int, help="For fixing random seed.")
    group.add_argument("--batch-size", default=64, type=int, help="Batch size during training.")
    group.add_argument(
        "--eval_during_training",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="to run a short (90 minutes) evaluation for each saved checkpoint.",
    )
    group.add_argument(
        "--val_during_training",
        nargs="?",
        const=True,
        default=True,
        type=str2bool,
        help="to run a short validation during training.",
    )
    group.add_argument("--short_db", action="store_true", help="Load short babel for debug.")
    group.add_argument("--cropping_sampler", action="store_true", help="Load short babel for debug.")
    group.add_argument("--data_path", default="./datasets/BEAT_SMPL/BEAT2/beat_english_v2.0.0/", help="add where the data is stored")
    group.add_argument("--data_amass_path", default="./datasets/AMASS-SMPLX/", help="add where the data of amass-smplx is stored")
    group.add_argument("--test_data_name", default=None, help="")
    group.add_argument("--custom_in_text_semantic", default=None, help="")
    group.add_argument("--static_pose_json", default=None, help="")
    group.add_argument(
        "--use_semantic_weighting", action="store_true", help="Enable semantic weighting for training based on beat velocity"
    )
    group.add_argument(
        "--do_not_use_clip",
        action="store_true",
        help="Use CLIP text encoder for text conditioning. Disable to run without CLIP.",
    )
    group.add_argument(
        "--rewrite_textgrid",
        action="store_true",
        help="Re-transcribe audio files with Whisper, saving new TextGrids to textgrid_whisper/ and transcripts to texts_whisper/.",
    )


def add_diffusion_options(parser):
    group = parser.add_argument_group("diffusion")
    group.add_argument(
        "--noise_schedule",
        default="cosine",
        choices=["linear", "cosine"],
        type=str,
        help="Noise schedule type",
    )
    group.add_argument(
        "--diffusion_steps",
        default=1000,
        type=int,
        help="Number of diffusion steps (denoted T in the paper)",
    )
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


# def add_diffusion_options_ddim(parser):
#     group = parser.add_argument_group("diffusion_ddim")

#     group.add_argument(
#         "--diffusion_steps",
#         default=1000,
#         type=int,
#         help="Number of diffusion steps (denoted T in the paper)",
#     )


def add_model_options(parser):
    group = parser.add_argument_group("model")
    group.add_argument(
        "--arch",
        default="trans_enc",
        choices=["trans_enc", "trans_dec", "gru"],
        type=str,
        help="Architecture types as reported in the paper.",
    )
    group.add_argument(
        "--emb_trans_dec",
        default=False,
        type=bool,
        help="For trans_dec architecture only, if true, will inject condition as a class token" " (in addition to cross-attention).",
    )
    group.add_argument("--layers", default=8, type=int, help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int, help="Transformer/GRU width.")
    group.add_argument(
        "--cond_mask_prob",
        default=0.1,
        type=float,
        help="The probability of masking the condition during training." " For classifier-free guidance learning.",
    )
    group.add_argument(
        "--cond_mask_prob_audio",
        default=0.15,
        type=float,
        help="The probability of masking the condition during training." " For classifier-free guidance learning.",
    )
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_rcxyz_hands", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument(
        "--lambda_clip_render", default=0.0, type=float, help="CLIP render loss (renders SMPLX mesh and compares with transcript text)."
    )
    group.add_argument(
        "--unconstrained",
        action="store_true",
        help="Model is trained unconditionally. That is, it is constrained by neither text nor action. " "Currently tested on HumanAct12 only.",
    )


def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--dataset",
        default="beat2",
        choices=["humanml", "kit", "humanact12", "uestc", "beat2"],
        type=str,
        help="Dataset name (choose from list).",
    )
    group.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="If empty, will use defaults according to the specified dataset.",
    )
    group.add_argument(
        "--use_amass",
        action="store_true",
        help="train with amass dataset",
    )


def add_training_options(parser):
    group = parser.add_argument_group("training")
    group.add_argument(
        "--save-dir",
        required=False,
        type=str,
        help="Path to save checkpoints and results.",
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="If True, will enable to use an already existing save-dir.",
    )
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument(
        "--lr-anneal-steps",
        default=0,
        type=int,
        help="Number of learning rate anneal steps.",
    )
    group.add_argument(
        "--eval-batch-size",
        default=32,
        type=int,
        help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
        "T2m precision calculation is based on fixed batch size 32.",
    )
    group.add_argument(
        "--eval-split",
        default="test",
        choices=["val", "test"],
        type=str,
        help="Which split to evaluate on during training.",
    )
    group.add_argument(
        "--eval-during-training",
        action="store_true",
        help="If True, will run evaluation during training.",
    )
    group.add_argument(
        "--eval-rep-times",
        default=3,
        type=int,
        help="Number of repetitions for evaluation loop during training.",
    )
    group.add_argument(
        "--eval-num-samples",
        default=1000,
        type=int,
        help="If -1, will use all samples in the specified split.",
    )
    group.add_argument("--log-interval", default=100, type=int, help="Log losses each N steps")
    group.add_argument(
        "--save-interval",
        default=50000,
        type=int,
        help="Save checkpoints and run evaluation each N steps",
    )
    group.add_argument(
        "--num_steps",
        default=600_000,
        type=int,
        help="Training will stop after the specified number of steps.",
    )
    group.add_argument(
        "--num_frames",
        default=60,
        type=int,
        help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.",
    )
    group.add_argument(
        "--resume_checkpoint",
        default="",
        type=str,
        help="If not empty, will start from the specified checkpoint (path to model###.pt file).",
    )
    group.add_argument(
        "--model-path",
        required=False,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )


def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--models-folder",
        required=False,
        type=str,
        help="Path to folderwith model####.pt files to be sampled.",
    )
    group.add_argument(
        "--models-names",
        required=False,
        type=str,
        default="",
        help="Path to folderwith model####.pt files to be sampled.",
    )
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Path to results dir (auto created by the script). " "If empty, will create dir in parallel to checkpoint.",
    )
    group.add_argument(
        "--num_samples",
        default=10,
        type=int,
        help="Maximal number of prompts to sample, " "if loading dataset from file, this field will be ignored.",
    )
    group.add_argument(
        "--num_repetitions",
        default=3,
        type=int,
        help="Number of repetitions, per sample (text prompt/action)",
    )
    group.add_argument(
        "--guidance_param",
        default=1,
        type=float,
        help="For classifier-free sampling - specifies the s parameter, as defined in the paper.",
    )


def add_double_take_options(parser):
    group = parser.add_argument_group("double_take")
    # group.add_argument("--double_take", action='store_true',
    #                    help="double take on the generated motion")
    group.add_argument("--double_take", default=True, type=bool, help="double take on the generated motion")
    group.add_argument("--second_take_only", action="store_true", help="double take on the generated motion")
    group.add_argument("--handshake_size", default=20, type=int, help="handshake size for unfolding")
    group.add_argument("--blend_len", default=20, type=int, help="blending with linear mask length")
    group.add_argument("--repaint_rep", default=10, type=int, help="number of times to sample during repaint")
    group.add_argument("--repaint", action="store_true", help="use repaint")
    group.add_argument("--debug_double_take", action="store_true", help="double_take debug mode")
    group.add_argument("--skip_steps_double_take", default=100, type=int, help="number of times to sample during repaint")


def add_seg_options(parser):
    group = parser.add_argument_group("seg")
    group.add_argument("--use_seg", action="store_true", help="inject semantic gestures from the SeG dataset")
    group.add_argument("-s", "--config_seg_opt_path", type=str, default=None)


def add_generate_options(parser):
    group = parser.add_argument_group("generate")
    group.add_argument(
        "--motion_length",
        default=6.0,
        type=float,
        help="The length of the sampled motion [in seconds]. "
        "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)",
    )
    group.add_argument(
        "--audio_path",
        default="",
        type=str,
        help="Path to an audio file to drive co-speech gesture generation (used by sample.inference).",
    )
    group.add_argument(
        "--model-path",
        required=False,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--input_text",
        default="",
        type=str,
        help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.",
    )
    group.add_argument(
        "--action_file",
        default="",
        type=str,
        help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
        "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
        "If no file is specified, will take action names from dataset.",
    )
    group.add_argument(
        "--text_prompt",
        default="",
        type=str,
        help="A text prompt to be generated. If empty, will take text prompts from dataset.",
    )
    group.add_argument(
        "--action_name",
        default="",
        type=str,
        help="An action name to be generated. If empty, will take text prompts from dataset.",
    )
    # group.add_argument(
    #     "--run_videos",
    #     default=False,
    #     required=False,
    #     type=bool,
    #     help="Genarate videos of the sampled motions.",
    # )
    group.add_argument("--run_videos", action="store_true", help="Run videos.")
    group.add_argument("--sample_gt", action="store_true", help="sample and visualize gt instead of generate sample")
    group.add_argument(
        "--person",
        default=None,
        type=str,
        help="Person identifier from ckp/official_model/ (e.g. '1_wayne' or just '1'). "
             "Auto-resolves --model-path and dataset cache_path.",
    )


def add_edit_options(parser):
    group = parser.add_argument_group("edit")
    group.add_argument(
        "--edit_mode",
        default="in_between",
        choices=["in_between", "upper_body"],
        type=str,
        help="Defines which parts of the input motion will be edited.\n"
        "(1) in_between - suffix and prefix motion taken from input motion, "
        "middle motion is generated.\n"
        "(2) upper_body - lower body joints taken from input motion, "
        "upper body is generated.",
    )
    group.add_argument(
        "--text_condition",
        default="",
        type=str,
        help="Editing will be conditioned on this text prompt. " "If empty, will perform unconditioned editing.",
    )
    group.add_argument(
        "--prefix_end",
        default=0.25,
        type=float,
        help="For in_between editing - Defines the end of input prefix (ratio from all frames).",
    )
    group.add_argument(
        "--suffix_start",
        default=0.75,
        type=float,
        help="For in_between editing - Defines the start of input suffix (ratio from all frames).",
    )


def add_evaluation_options(parser):
    group = parser.add_argument_group("eval")
    group.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--eval_mode",
        default="wo_mm",
        choices=["wo_mm", "mm_short", "debug", "full"],
        type=str,
        help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
        "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
        "debug - short run, less accurate results."
        "full (a2m only) - 20 repetitions.",
    )
    group.add_argument(
        "--guidance_param",
        default=1,
        type=float,
        help="For classifier-free sampling - specifies the s parameter, as defined in the paper.",
    )


def add_train_platform_options(parser):
    group = parser.add_argument_group("train platform")
    group.add_argument(
        "--train-platform-type",
        default="NoPlatform",
        choices=[
            "NoPlatform",
            "ClearmlPlatform",
            "TensorboardPlatform",
            "WandbPlatform",
        ],
        type=str,
        help="Choose platform to log results. NoPlatform means no logging.",
    )
    group.add_argument("--entity", type=str, help="Wandb entity.")
    group.add_argument("--project", type=str, help="Wandb project.")


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = "no_cond"
    elif args.dataset in ["kit", "humanml", "beat2"]:
        cond_mode = "text"
    else:
        cond_mode = "action"
    return cond_mode


def add_frame_sampler_options(parser):
    group = parser.add_argument_group("framesampler")
    group.add_argument("--min_seq_len", default=45, type=int, help="babel dataset FrameSampler minimum length")
    group.add_argument("--max_seq_len", default=250, type=int, help="babel dataset FrameSampler maximum length")


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_train_platform_options(parser)
    add_seg_options(parser)
    args, unknown = parser.parse_known_args()
    return args


def generate_args(model_path=None):
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_frame_sampler_options(parser)
    add_double_take_options(parser)
    add_seg_options(parser)

    if model_path is None:
        pre_args, _ = parser.parse_known_args()
        if getattr(pre_args, "person", None):
            model_path = _resolve_person_model(pre_args.person)

    args = parse_and_load_from_model(parser, model_path)

    if model_path is not None and not getattr(args, "model_path", None):
        args.model_path = model_path

    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != "text":
        raise Exception(
            "Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name."
        )
    elif (args.action_file or args.action_name) and cond_mode != "action":
        raise Exception(
            "Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt."
        )

    return args


def evaluate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != "text":
        raise Exception(
            "Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name."
        )
    elif (args.action_file or args.action_name) and cond_mode != "action":
        raise Exception(
            "Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt."
        )

    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)

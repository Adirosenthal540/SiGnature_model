from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
from model.cfg_sampler import wrap_model
import torch
from model.cfg_sampler import ClassifierFreeSampleModel


def load_model_wo_clip(model, state_dict):
    # prefix = "base_model.model."
    # for k in list(state_dict.keys()):   # snapshot -> safe
    #     if k.startswith(prefix):
    #         new_k = k[len(prefix):]
    #         state_dict[new_k] = state_dict.pop(k)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all(["clip_model." in k or "film." in k for k in missing_keys])


def load_model(args, device, ModelClass=MDM, model_path=None):
    model, diffusion = create_model_and_diffusion(args, ModelClass=ModelClass)
    if model_path is None:
        model_path = args.model_path
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location="cpu")

    load_model_wo_clip(model, state_dict)

    model.to(device)
    model.eval()  # disable random masking
    model = wrap_model(model, args.guidance_param)
    return model, diffusion


def create_model_and_diffusion(args, ModelClass=MDM, DiffusionClass=SpacedDiffusion, do_not_use_clip=False):
    model = ModelClass(**get_model_args(args), do_not_use_clip=args.do_not_use_clip)
    diffusion = create_gaussian_diffusion(args, DiffusionClass)
    return model, diffusion


def get_model_args(args):

    # default args
    clip_version = "ViT-B/32"
    action_emb = "tensor"
    cond_mode = get_cond_mode(args)
    njoints = 337  # the size of latet all - todo try change to 333
    data_rep = "hml_vec"
    nfeats = 1

    return {
        "modeltype": "",
        "njoints": njoints,
        "nfeats": nfeats,
        "translation": True,
        "pose_rep": "rot6d",
        "glob": True,
        "glob_rot": True,
        "latent_dim": args.latent_dim,
        "ff_size": 512,
        "num_layers": args.layers,
        "num_heads": 4,
        "dropout": 0.1,
        "activation": "gelu",
        "data_rep": data_rep,
        "cond_mode": cond_mode,
        "cond_mask_prob": args.cond_mask_prob,
        "action_emb": action_emb,
        "arch": args.arch,
        "emb_trans_dec": args.emb_trans_dec,
        "clip_version": clip_version,
        "dataset": args.dataset,
        "device": args.device,
        "data_path": args.data_path,
        "cond_mask_prob_audio": args.cond_mask_prob_audio,
    }


def create_gaussian_diffusion(args, DiffusionClass=SpacedDiffusion):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.0  # no scaling
    timestep_respacing = ""  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if hasattr(args, "multi_train_mode"):
        multi_train_mode = args.multi_train_mode
    else:
        multi_train_mode = None

    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not args.sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_rcxyz_hands=args.lambda_rcxyz_hands,
        lambda_fc=args.lambda_fc,
        lambda_clip_render=getattr(args, "lambda_clip_render", 0.0),
        batch_size=args.batch_size,
        multi_train_mode=multi_train_mode,
        device=f"cuda:{getattr(args, 'device', 0)}" if isinstance(getattr(args, "device", 0), int) else getattr(args, "device", "cuda"),
    )

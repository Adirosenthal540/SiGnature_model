# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import os
import random
import numpy as np
import torch
import torch as th
import copy
from copy import deepcopy
import torch.nn.functional as F
from data_loaders.tensors import beat2_collate
from utils import rotation_conversions as rc
from diffusion.nn import mean_flat, sum_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood



def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.0):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        lambda_rcxyz=0.0,
        lambda_rcxyz_hands=0.0,
        lambda_vel=0.0,
        lambda_pose=1.0,
        lambda_orient=1.0,
        lambda_loc=1.0,
        data_rep="rot6d",
        lambda_root_vel=0.0,
        lambda_vel_rcxyz=0.0,
        lambda_fc=0.0,
        lambda_clip_render=0.0,
        batch_size=32,
        lambda_semantic_weighting=1,
        multi_train_mode=None,
        device="cuda",
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep
        self.multi_train_mode = multi_train_mode

        if data_rep != "rot_vel" and lambda_pose != 1.0:
            raise ValueError("lambda_pose is relevant only when training on velocities!")
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_rcxyz_hands = lambda_rcxyz_hands
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc
        self.lambda_clip_render = lambda_clip_render
        self.lambda_semantic = 1000
        self.lambda_semantic_weighting = lambda_semantic_weighting
        self.device = device

        if (
            self.lambda_rcxyz > 0.0
            or self.lambda_vel > 0.0
            or self.lambda_root_vel > 0.0
            or self.lambda_vel_rcxyz > 0.0
            or self.lambda_fc > 0.0
            or self.lambda_clip_render > 0.0
        ):
            assert self.loss_type == LossType.MSE, "Geometric losses are supported by MSE loss type only!"

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

        self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.
        self.cache_dict = {}
        # if self.lambda_rcxyz or self.lambda_vel_rcxyz or self.lambda_fc or self.lambda_vel:  # TODO - for Babel dataset only
        #     self.transform = SlimSMPLTransform(
        #         batch_size=batch_size, name="SlimSMPLTransform", ename="smplnh", normalization=True
        #     )  # data_loader.dataset.transform
        #     self.Datastruct = self.transform.SlimDatastruct

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        non_zero_elements[non_zero_elements == 0] = 1
        mse_loss_val = loss / non_zero_elements

        return mse_loss_val

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, model_kwargs=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        if self.multi_train_mode == "prefix":
            bs, feat, _, frames = noise.shape
            prefix_size = 20  # FIXME - HARDCODED for the pw3d task
            inpainting_mask = torch.zeros_like(noise)
            inpainting_mask[..., : prefix_size + 1] = 1.0  #  +1 for 6dof
            noise *= 1.0 - inpainting_mask
            return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
        else:
            return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        assert posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, insert_sg_info_with_a_pose=None, seg_dataset=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C, _, length_max = x.shape
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if insert_sg_info_with_a_pose is not None and insert_sg_info_with_a_pose != []:
            gestures = insert_sg_info_with_a_pose["gestures"]
            action = insert_sg_info_with_a_pose["action"]
            buffer_seg = 45
            pre_frames = 30
            # if self.num_timesteps - t[0] <= seg_dataset.guidance_T:
            for index_seg, seg_motion_info in enumerate(gestures):
                for round in range(model_output.shape[0]):
                    if "frames_indexs" not in model_kwargs["y"].keys():
                        range_indxs = [0, length_max]
                    else:
                        range_indxs = model_kwargs["y"]["frames_indexs"][round]
                    start_frame = seg_motion_info["start_code"]
                    end_frame = seg_motion_info["end_code"]

                    static_pose = seg_motion_info.get("static_pose", False)
                    lambda_integration = seg_motion_info.get("lambda_integration", 0.7)
                    blend_frames = seg_motion_info.get("blend_frames", 20)
                    if not static_pose:
                        middle_frame = int((start_frame + end_frame) / 2)
                        shift = 10
                        start_frame = middle_frame - buffer_seg + shift
                        end_frame = middle_frame + buffer_seg + shift
                    if (start_frame < (range_indxs[0] + pre_frames) and end_frame < (range_indxs[0] + pre_frames)) or (
                        start_frame > (range_indxs[1] - pre_frames) and end_frame > (range_indxs[1] - pre_frames)
                    ):
                        continue  # todo - check if this is correct - why pre_frames needed?

                    choice_index = seg_motion_info["choice_index"]
                    pose_gesture = seg_motion_info["semantic_gesture_info"][choice_index]["poses_6d"].to(x.device).permute(0, 2, 1).unsqueeze(2)

                    max_length = model_output.shape[-1]
                    name_motion = seg_motion_info["semantic_gesture_info"][choice_index]["file_name"][:-4]

                    replace_s_index_orig = int((start_frame - range_indxs[0]))
                    shift_s = 0
                    if replace_s_index_orig < 0:
                        shift_s = np.abs(replace_s_index_orig)
                        replace_s_index = 0
                    else:
                        replace_s_index = replace_s_index_orig

                    shift_e = 0
                    replace_e_index_orig = int(np.ceil((end_frame - range_indxs[0])))
                    if replace_e_index_orig > max_length:
                        shift_e = replace_e_index_orig - max_length
                        replace_e_index = max_length
                    else:
                        replace_e_index = replace_e_index_orig

                    length_pose = end_frame - start_frame

                    _target = pose_gesture[:, :, :, shift_s : (length_pose - shift_e)]
                    alpha = (torch.arange(0, blend_frames + 1, 1, device=_target.device) / (blend_frames + 1))[1:]
                    mask_sm = torch.ones(length_pose).to(_target.device)
                    if blend_frames > 0:
                        mask_sm[:blend_frames] = alpha
                        mask_sm[-blend_frames:] = 1 - alpha  # todo - can move to global
                    mask_crop = mask_sm[shift_s : (length_pose - shift_e)]
                    if action == "edit":
                        joints_rot6d_mask = seg_motion_info["joints_rot6d_mask"]
                        # joints_rot6d_mask = seg_motion_info["joints_rot6d_mask"]
                        mask_crop *= lambda_integration

                    elif action == "transfer":
                        joints_rot6d_mask = torch.ones(330, dtype=bool).to(x.device)
                    else:
                        raise ValueError(f"Invalid action: {action}")

                    blend = (
                        model_output[round, :330, :, replace_s_index:replace_e_index][joints_rot6d_mask, :, :] * (1 - mask_crop)  # .reshape(1,1,-1)
                        + _target[0][:330][joints_rot6d_mask, :, :] * mask_crop
                    )
                    if seg_motion_info.get("use_global_pose", False):
                        model_output[:, 330:333, :, :] = (
                            model_output[0, 330:333, :, replace_s_index].repeat((1, 1, 196)).repeat((model_output.shape[0], 1, 1)).unsqueeze(2)
                        )

                    model_output[round, :330, :, replace_s_index:replace_e_index][joints_rot6d_mask, :, :] = blend.to(model_output.dtype)

        model_variance, model_log_variance = {
            # for fixedlarge, set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(model_output)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, p_mean_var, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        insert_sg_info_with_a_pose=None,
        seg_dataset=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            insert_sg_info_with_a_pose=insert_sg_info_with_a_pose,
            seg_dataset=seg_dataset,
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        predict_two_person=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        unfolding_handshake=0,  # 0 means no unfolding
        repaint_samples=1,  # 1 means no repaint
        arb_len=False,
        second_take_only=False,
        seg_dataset=None,
        insert_sg_info_with_a_pose=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(
            self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                predict_two_person=predict_two_person,
                skip_timesteps=skip_timesteps,
                init_image=init_image,
                randomize_class=randomize_class,
                cond_fn_with_grad=cond_fn_with_grad,
                const_noise=const_noise,
                seg_dataset=seg_dataset,
                insert_sg_info_with_a_pose=insert_sg_info_with_a_pose,
            )
        ):

            # unfolding
            if (arb_len) and (unfolding_handshake > 0) and not (second_take_only):
                alpha = torch.arange(0, unfolding_handshake, 1, device=sample["sample"].device) / unfolding_handshake
                for sample_i, length in zip(range(1, sample["sample"].shape[0]), model_kwargs["y"]["lengths"]):
                    _suffix = sample["sample"][sample_i - 1, :, :, -unfolding_handshake + length : length]
                    _prefix = sample["sample"][sample_i, :, :, :unfolding_handshake]
                    try:
                        _blend = _suffix * (1 - alpha) + _prefix * alpha
                    except RuntimeError:
                        print("Error")
                    sample["sample"][sample_i - 1, :, :, -unfolding_handshake + length : length] = _blend
                    sample["sample"][sample_i, :, :, :unfolding_handshake] = _blend
            elif (unfolding_handshake > 0) and not (second_take_only):
                for sample_i in range(1, sample["sample"].shape[0]):
                    _suffix = sample["sample"][sample_i - 1, :, :, -unfolding_handshake:]
                    _prefix = sample["sample"][sample_i, :, :, :unfolding_handshake]
                    _blend = _suffix * (1 - alpha) + _prefix * alpha
                    sample["sample"][sample_i - 1, :, :, -unfolding_handshake:] = _blend
                    sample["sample"][sample_i, :, :, :unfolding_handshake] = _blend
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))

            final = sample
        if dump_steps is not None:
            return dump
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        predict_two_person=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
        insert_sg_info_with_a_pose=None,
        seg_dataset=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            if predict_two_person:
                img = [th.randn(*shape, device=device), th.randn(*shape, device=device)]
            else:
                img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            if predict_two_person:
                img[0] = self.q_sample(init_image[0].to(device), my_t, img[0], model_kwargs=model_kwargs)
                img[1] = self.q_sample(init_image[1].to(device), my_t, img[1], model_kwargs=model_kwargs)
            else:
                img = self.q_sample(init_image, my_t, img, model_kwargs=model_kwargs)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and "y" in model_kwargs:
                model_kwargs["y"] = th.randint(low=0, high=model.num_classes, size=model_kwargs["y"].shape, device=model_kwargs["y"].device)
            if 1:
                # with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                if predict_two_person:
                    sample_fn = self.p_sample_multi
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                    insert_sg_info_with_a_pose=insert_sg_info_with_a_pose,
                    seg_dataset=seg_dataset,
                )
                yield out
                img = out["sample"]

    def p_sample_multi(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """

        x1, x2 = x

        if "inpainted_motion_multi" in model_kwargs["y"].keys():
            model_kwargs["y"]["inpainted_motion"] = model_kwargs["y"]["inpainted_motion_multi"][0]

        model_kwargs["y"]["other_motion"] = x2
        out1 = self.p_mean_variance(
            model,
            x1,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        if "inpainted_motion_multi" in model_kwargs["y"].keys():
            model_kwargs["y"]["inpainted_motion"] = model_kwargs["y"]["inpainted_motion_multi"][1]

        model_kwargs["y"]["other_motion"] = x1
        out2 = self.p_mean_variance(
            model,
            x2,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        def handle_sample(_x, _out, _cond_fn, _t, _const_noise):
            noise = th.randn_like(_x)
            if _const_noise:
                noise = noise[[0]].repeat(_x.shape[0], 1, 1, 1)

            nonzero_mask = (_t != 0).float().view(-1, *([1] * (len(_x.shape) - 1)))  # no noise when t == 0
            if _cond_fn is not None:
                _out["mean"] = self.condition_mean(_cond_fn, _out, _x, _t, model_kwargs=model_kwargs)
            if self.multi_train_mode == "prefix":
                assert "inpainting_mask" in model_kwargs["y"].keys()
                # FOR FINETUNED MODELS ONLY !!
                inpainting_mask = model_kwargs["y"]["inpainting_mask"].to(noise.device)
                inpainting_mask = inpainting_mask.float()
                noise *= 1.0 - inpainting_mask
            _sample = _out["mean"] + nonzero_mask * th.exp(0.5 * _out["log_variance"]) * noise
            return _sample

        sample1 = handle_sample(x1, out1, cond_fn, t, const_noise)
        sample2 = handle_sample(x2, out2, cond_fn, t, const_noise)
        sample = (sample1, sample2)
        out = (out1, out2)

        return {"sample": sample, "pred_xstart": (out1["pred_xstart"], out2["pred_xstart"])}

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        seg_dataset=None,
        insert_sg_info_with_a_pose=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            insert_sg_info_with_a_pose=insert_sg_info_with_a_pose,
            seg_dataset=seg_dataset,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_reverse_sample_loop(
        self,
        model,
        x,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        progress=False,
        eta=0.0,
        device=None,
    ):

        if device is None:
            device = next(model.parameters()).device
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(self.num_timesteps))
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        sample = x
        for i in indices:
            t = th.tensor([i] * len(sample), device=device)
            with th.no_grad():
                out = self.ddim_optimize_reverse_sample(
                    model,
                    sample,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    num_iter=2,
                )
                sample = out["sample"]
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out["pred_xstart"])
                # [0, ..., T-1] ready to use
                T.append(t)

        return {
            #  xT "
            "sample": sample,
            # (1, ..., T)
            "sample_t": sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            "xstart_t": xstart_t,
            "T": T,
        }

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_xstart"]) / _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x.shape
        )
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_optimize_reverse_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0, num_iter=100):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"

        last = x.clone()
        # latent_0 = x.clone()
        for i in range(num_iter):
            out = self.p_mean_variance(
                model,
                last,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, last.shape) * x - out["pred_xstart"]) / _extract_into_tensor(
                self.sqrt_recipm1_alphas_cumprod, t, last.shape
            )
            alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, last.shape)

            # Equation 12. reversed
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps
            latent = mean_pred
            score = torch.norm(last - latent)
            last = latent.clone()
        print("score: ", score.item())
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        unfolding_handshake=0,  # 0 means no unfolding
        repaint_samples=1,  # 1 means no repaint
        arb_len=False,
        second_take_only=False,
        seg_dataset=None,
        insert_sg_info_with_a_pose=None,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        if noise is None:
            noise = th.randn(*shape, device=device)

        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            seg_dataset=seg_dataset,
            insert_sg_info_with_a_pose=insert_sg_info_with_a_pose,
        ):

            # unfolding
            if (arb_len) and (unfolding_handshake > 0) and not (second_take_only):
                alpha = torch.arange(0, unfolding_handshake, 1, device=sample["sample"].device) / unfolding_handshake
                for sample_i, length in zip(range(1, sample["sample"].shape[0]), model_kwargs["y"]["lengths"]):
                    _suffix = sample["sample"][sample_i - 1, :, :, -unfolding_handshake + length : length]
                    _prefix = sample["sample"][sample_i, :, :, :unfolding_handshake]
                    try:
                        _blend = _suffix * (1 - alpha) + _prefix * alpha
                    except RuntimeError:
                        print("Error")
                    sample["sample"][sample_i - 1, :, :, -unfolding_handshake + length : length] = _blend
                    sample["sample"][sample_i, :, :, :unfolding_handshake] = _blend
            elif (unfolding_handshake > 0) and not (second_take_only):
                for sample_i in range(1, sample["sample"].shape[0]):
                    _suffix = sample["sample"][sample_i - 1, :, :, -unfolding_handshake:]
                    _prefix = sample["sample"][sample_i, :, :, :unfolding_handshake]
                    _blend = _suffix * (1 - alpha) + _prefix * alpha
                    sample["sample"][sample_i - 1, :, :, -unfolding_handshake:] = _blend
                    sample["sample"][sample_i, :, :, :unfolding_handshake] = _blend
            # if dump_steps is not None and i in dump_steps:
            #     dump.append(deepcopy(sample["sample"]))
            final = sample
        return final["sample"], noise

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        seg_dataset=None,
        insert_sg_info_with_a_pose=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img, model_kwargs=model_kwargs)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and "y" in model_kwargs:
                model_kwargs["y"] = th.randint(low=0, high=model.num_classes, size=model_kwargs["y"].shape, device=model_kwargs["y"].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    insert_sg_info_with_a_pose=insert_sg_info_with_a_pose,
                    seg_dataset=seg_dataset,
                )
                yield out
                img = out["sample"]

    def plms_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cond_fn_with_grad=False,
        order=2,
        old_out=None,
    ):
        """
        Sample x_{t-1} from the model using Pseudo Linear Multistep.

        Same usage as p_sample().
        """
        if not int(order) or not 1 <= order <= 4:
            raise ValueError("order is invalid (should be int from 1-4).")

        def get_model_output(x, t):
            with th.set_grad_enabled(cond_fn_with_grad and cond_fn is not None):
                x = x.detach().requires_grad_() if cond_fn_with_grad else x
                out_orig = self.p_mean_variance(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if cond_fn is not None:
                    if cond_fn_with_grad:
                        out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                        x = x.detach()
                    else:
                        out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                else:
                    out = out_orig

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            return eps, out, out_orig

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        eps, out, out_orig = get_model_output(x, t)

        if order > 1 and old_out is None:
            # Pseudo Improved Euler
            old_eps = [eps]
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps
            eps_2, _, _ = get_model_output(mean_pred, t - 1)
            eps_prime = (eps + eps_2) / 2
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime
        else:
            # Pseudo Linear Multistep (Adams-Bashforth)
            old_eps = old_out["old_eps"]
            old_eps.append(eps)
            cur_order = min(order, len(old_eps))
            if cur_order == 1:
                eps_prime = old_eps[-1]
            elif cur_order == 2:
                eps_prime = (3 * old_eps[-1] - old_eps[-2]) / 2
            elif cur_order == 3:
                eps_prime = (23 * old_eps[-1] - 16 * old_eps[-2] + 5 * old_eps[-3]) / 12
            elif cur_order == 4:
                eps_prime = (55 * old_eps[-1] - 59 * old_eps[-2] + 37 * old_eps[-3] - 9 * old_eps[-4]) / 24
            else:
                raise RuntimeError("cur_order is invalid.")
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime

        if len(old_eps) >= order:
            old_eps.pop(0)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred * nonzero_mask + out["pred_xstart"] * (1 - nonzero_mask)

        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "old_eps": old_eps}

    def plms_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Generate samples from the model using Pseudo Linear Multistep.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.plms_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            order=order,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from each
        timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img, model_kwargs=model_kwargs)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_out = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and "y" in model_kwargs:
                model_kwargs["y"] = th.randint(low=0, high=model.num_classes, size=model_kwargs["y"].shape, device=model_kwargs["y"].device)
            with th.no_grad():
                out = self.plms_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cond_fn_with_grad=cond_fn_with_grad,
                    order=order,
                    old_out=old_out,
                )
                yield out
                old_out = out
                img = out["sample"]

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        # enc = model.model._modules['module']
        enc = model.model
        mask = model_kwargs["y"]["mask"]

        get_xyz = lambda sample: enc.rot2xyz(
            sample,
            mask=None,
            pose_rep=enc.pose_rep,
            translation=enc.translation,
            glob=enc.glob,
            # jointstype='vertices',  # 3.4 iter/sec # USED ALSO IN MotionCLIP
            jointstype="smpl",  # 3.4 iter/sec
            vertstrans=False,
        )

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise, model_kwargs=model_kwargs)

        if hasattr(model.model, "is_multi") and model.model.is_multi:
            noise2 = th.randn_like(x_start)
            model_kwargs["y"]["other_motion"] = self.q_sample(model_kwargs["y"]["other_motion"], t, noise=noise2, model_kwargs=model_kwargs)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = x_start
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]

            target_poses = target[:, :330]
            target_trans = target[:, 330:333]
            target_fc = target[:, 333:]
            model_output_poses = model_output[:, :330]
            model_output_trans = model_output[:, 330:333]
            # model_output_fc = model_output[333:]
            terms["rot_mse"] = self.masked_l2(target_poses, model_output_poses, mask)  # mean_flat(rot_mse)
            terms["trans_mse"] = self.masked_l2(target_trans, model_output_trans, mask)  # mean_flat(trans_mse)

            target_xyz, model_output_xyz = None, None

            if self.lambda_rcxyz > 0.0:
                # target_xyz = get_xyz(target)  # [bs, nvertices(vertices)/njoints(smpl), 3, nframes]
                # model_output_xyz = get_xyz(model_output)  # [bs, nvertices, 3, nframes]

                if "target_xyz" not in model_kwargs["y"]:
                    target_xyz = enc.rot2xyz(target, "rot_6d", data=dataset).permute(0, 2, 3, 1).to(target.device)
                    model_kwargs["y"]["target_xyz"] = target_xyz
                else:
                    target_xyz = model_kwargs["y"]["target_xyz"]

                model_output_xyz = enc.rot2xyz(model_output, "rot_6d", data=dataset).permute(0, 2, 3, 1).to(target.device)
                terms["rcxyz_mse"] = self.masked_l2(target_xyz, model_output_xyz, mask)  # mean_flat((target_xyz - model_output_xyz) ** 2)

            if 0:  # self.lambda_rcxyz_hands > 0.0:
                # left_elbow, right_elbow, left_wrist, right_wrist = 18, 19, 20, 21
                index_hands = dataset.joint_mask_hands.reshape(55, 3).any(axis=1)
                joints_rot6d_mask = np.repeat(index_hands, 6)
                indexs = np.where(joints_rot6d_mask)[0].tolist()
                terms["mse_hands"] = self.masked_l2(
                    target[:, indexs], model_output[:, indexs], mask
                )  # mean_flat((target_xyz - model_output_xyz) ** 2)
                # terms["rot_mse"] = 0
                # index_hands = np.where(dataset.joint_mask_hands.reshape(55, 3).any(axis=1))[0].tolist()
                # if "target_xyz_hands" not in model_kwargs["y"]:
                #     target_xyz = (
                #         enc.rot2xyz(target, "rot_6d", data=dataset, joint_mapper=torch.LongTensor(index_hands)).permute(0, 2, 3, 1).to(target.device)
                #     )
                #     model_kwargs["y"]["target_xyz_hands"] = target_xyz
                # else:
                #     target_xyz = model_kwargs["y"]["target_xyz_hands"]

                # model_output_xyz = (
                #     enc.rot2xyz(model_output, "rot_6d", data=dataset, joint_mapper=torch.LongTensor(index_hands))
                #     .permute(0, 2, 3, 1)
                #     .to(target.device)
                # )
                # terms["rcxyz_mse_hands"] = self.masked_l2(target_xyz[:, index_hands], model_output_xyz[:, index_hands], mask)  # mean_flat((target_xyz - model_output_xyz) ** 2)

            if self.lambda_vel_rcxyz > 0.0:
                if self.data_rep == "rot6d" and dataset.dataname in ["humanact12", "uestc", "BABEL", "babel"]:
                    # target_xyz = get_xyz(target) if target_xyz is None else target_xyz
                    # model_output_xyz = get_xyz(model_output) if model_output_xyz is None else model_output_xyz
                    target_xyz_vel = target_xyz[:, :, :, 1:] - target_xyz[:, :, :, :-1]
                    model_output_xyz_vel = model_output_xyz[:, :, :, 1:] - model_output_xyz[:, :, :, :-1]
                    terms["vel_xyz_mse"] = self.masked_l2(target_xyz_vel, model_output_xyz_vel, mask[:, :, :, 1:])

            if self.lambda_fc > 0.0 and dataset is not None:

                # torch.autograd.set_detect_anomaly(True)
                # self.data_rep == "rot6d" and dataset.dataname in ["humanact12", "uestc", "BABEL", "babel"]:
                l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11

                relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]  # [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
                if "target_xyz" not in model_kwargs["y"]:
                    target_xyz = (
                        enc.rot2xyz(target, "rot_6d", data=dataset, joint_mapper=torch.LongTensor([l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]))
                        .permute(0, 2, 3, 1)
                        .to(target.device)
                    )
                    model_kwargs["y"]["target_xyz"] = target_xyz
                else:
                    target_xyz = model_kwargs["y"]["target_xyz"]
                if model_output_xyz is None:
                    model_output_xyz = (
                        enc.rot2xyz(
                            model_output,
                            "rot_6d",
                            data=dataset,  # , joint_mapper=torch.LongTensor([l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx])
                        )
                        .permute(0, 2, 3, 1)
                        .to(target.device)
                    )

                # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11

                gt_joint_xyz = target_xyz[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
                gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
                fc_mask = torch.unsqueeze((gt_joint_vel < 0.01), dim=2).repeat(1, 1, 3, 1)
                # target_fc.repeat(1, 1, 3, 1)

                pred_joint_xyz = model_output_xyz[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
                pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
                pred_vel[~fc_mask] = 0
                # pred_vel[~target_fc.squeeze(2)[:, :, :-1].bool().unsqueeze(2).repeat(1, 1, 3, 1)] = 0
                terms["fc"] = self.masked_l2(pred_vel, torch.zeros(pred_vel.shape, device=pred_vel.device), mask[:, :, :, :-1])

            # if self.lambda_semantic > 0.0 and dataset is not None:

            #     # torch.autograd.set_detect_anomaly(True)
            #     # self.data_rep == "rot6d" and dataset.dataname in ["humanact12", "uestc", "BABEL", "babel"]:
            #     upper_body_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            #     if "target_xyz" not in model_kwargs["y"]:
            #         target_xyz = enc.rot2xyz(target, "rot_6d", data=dataset).permute(0, 2, 3, 1).to(target.device)
            #         model_kwargs["y"]["target_xyz"] = target_xyz
            #     else:
            #         target_xyz = model_kwargs["y"]["target_xyz"]
            #     if model_output_xyz is None:
            #         model_output_xyz = enc.rot2xyz(model_output, "rot_6d", data=dataset).permute(0, 2, 3, 1).to(target.device)
            #     gt_joint_xyz = target_xyz[:, :, :, :]
            #     gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2) / torch.tensor(
            #         dataset.avg_vel[:], device=target.device
            #     ).reshape(1, -1, 1).repeat(1, 1, gt_joint_xyz.shape[3] - 1)
            #     fc_mask = gt_joint_vel > 0.3  # .repeat(1, 1, 3, 1)
            #     pred_joint_xyz = model_output_xyz[:, :, :, :]
            #     pred_vel = torch.linalg.norm(pred_joint_xyz[:, :, :, :1] - pred_joint_xyz[:, :, :, :-1], axis=2) / torch.tensor(
            #         dataset.avg_vel[:], device=target.device
            #     ).reshape(1, -1, 1).repeat(1, 1, pred_joint_xyz.shape[3] - 1)
            #     # gt_joint_xyz = target_xyz[:, upper_body_joints, :, :]
            #     # gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2) / torch.tensor(
            #     #     dataset.avg_vel[upper_body_joints], device=target.device
            #     # ).reshape(1, -1, 1).repeat(1, 1, gt_joint_xyz.shape[3] - 1)
            #     # fc_mask = gt_joint_vel > 0.3  # .repeat(1, 1, 3, 1)
            #     # pred_joint_xyz = model_output_xyz[:, upper_body_joints, :, :]
            #     # pred_vel = torch.linalg.norm(pred_joint_xyz[:, :, :, :1] - pred_joint_xyz[:, :, :, :-1], axis=2) / torch.tensor(
            #     #     dataset.avg_vel[upper_body_joints], device=target.device
            #     # ).reshape(1, -1, 1).repeat(1, 1, pred_joint_xyz.shape[3] - 1)
            #     pred_vel[~fc_mask] = 0
            #     gt_joint_vel[~fc_mask] = 0
            #     terms["semantic"] = self.masked_l2(gt_joint_vel, pred_vel, mask[:, :, 0, 1:])
            if self.lambda_vel > 0.0:
                target_vel = target[..., 1:] - target[..., :-1]
                model_output_vel = model_output[..., 1:] - model_output[..., :-1]
                terms["vel_mse"] = self.masked_l2(
                    target_vel[:, :-1, :, :], model_output_vel[:, :-1, :, :], mask[:, :, :, 1:]  # Remove last joint, is the root location!
                )  # mean_flat((target_vel - model_output_vel) ** 2)

            # # Apply semantic weighting if enabled and dataset is available
            # if hasattr(self, "semantic_weighting") and self.semantic_weighting is not None and dataset is not None:
            #     try:
            #         # Get GT motion for semantic weighting calculation
            #         if "target_xyz" not in model_kwargs["y"]:
            #             target_xyz = (
            #                 enc.rot2xyz(
            #                     target, "rot_6d", data=dataset, joint_mapper=torch.LongTensor([l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx])
            #                 )
            #                 .permute(0, 2, 3, 1)
            #                 .to(target.device)
            #             )
            #             model_kwargs["y"]["target_xyz"] = target_xyz
            #         else:
            #             target_xyz = model_kwargs["y"]["target_xyz"]
            #         # Apply semantic weighting to loss terms
            #         weighted_terms = self.apply_semantic_weighting(terms, target_xyz, dataset)
            terms["loss"] = (
                terms["rot_mse"]
                + terms["trans_mse"]
                + terms.get("vb", 0.0)
                + (self.lambda_vel * terms.get("vel_mse", 0.0))
                + (self.lambda_rcxyz * terms.get("rcxyz_mse", 0.0))
                + (self.lambda_fc * terms.get("fc", 0.0))
            )

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def set_semantic_weighting(self, semantic_weighting):
        """
        Set semantic weighting module for training.

        Args:
            semantic_weighting: SemanticWeighting object
        """
        self.semantic_weighting = semantic_weighting

    def apply_semantic_weighting(self, loss_terms, gt_xyz, dataset):
        """
        Apply semantic weighting to loss terms.

        Args:
            loss_terms: Dictionary of loss terms
            gt_xyz: Ground truth joint positions [batch_size, n_joints, 3, n_frames]
            dataset: Dataset object

        Returns:
            Dictionary with weighted loss terms
        """
        if self.semantic_weighting is None:
            return {}

        try:
            # Convert tensor to numpy for alignmenter
            gt_xyz_np = gt_xyz.detach().cpu().numpy().transpose(0, 3, 1, 2)  # [batch_size, n_frames, n_joints, 3]

            weighted_terms = {}
            batch_size = gt_xyz_np.shape[0]

            # Apply semantic weighting to each sample in the batch
            for i in range(batch_size):
                joints_body = gt_xyz_np[i]  # [n_frames, n_joints, 3]

                # Create semantic weights for this sample
                semantic_weights = self.semantic_weighting.create_semantic_weights(joints_body, align_mask=0, pose_fps=30, semantic_threshold=0.5)

                # Apply weights to loss terms
                for key, loss_term in loss_terms.items():
                    if key not in weighted_terms:
                        weighted_terms[key] = []

                    # Apply semantic weights to this loss term
                    if loss_term.dim() > 1 and loss_term.shape[1] == semantic_weights.shape[0]:
                        # Frame-wise weighting
                        weighted_loss = loss_term[i] * semantic_weights
                        weighted_terms[key].append(weighted_loss)
                    else:
                        # No weighting applied
                        weighted_terms[key].append(loss_term[i])

            # Stack weighted terms back to batch format
            for key in weighted_terms:
                weighted_terms[key] = torch.stack(weighted_terms[key], dim=0)

            return weighted_terms

        except Exception as e:
            print(f"Error in semantic weighting: {e}")
            return {}

    def fc_loss_rot_repr(self, gt_xyz, pred_xyz, mask):
        def to_np_cpu(x):
            return x.detach().cpu().numpy()

        """
        pose_xyz: SMPL batch tensor of shape: [BatchSize, 24, 3, Frames]
        """
        # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11

        l_ankle_idx, r_ankle_idx = 7, 8
        l_foot_idx, r_foot_idx = 10, 11
        """ Contact calculated by 'Kfir Method' Commented code)"""
        # contact_signal = torch.zeros((pose_xyz.shape[0], pose_xyz.shape[3], 2), device=pose_xyz.device) # [BatchSize, Frames, 2]
        # left_xyz = 0.5 * (pose_xyz[:, l_ankle_idx, :, :] + pose_xyz[:, l_foot_idx, :, :]) # [BatchSize, 3, Frames]
        # right_xyz = 0.5 * (pose_xyz[:, r_ankle_idx, :, :] + pose_xyz[:, r_foot_idx, :, :])
        # left_z, right_z = left_xyz[:, 2, :], right_xyz[:, 2, :] # [BatchSize, Frames]
        # left_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)  # [BatchSize, Frames]
        # right_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)
        #
        # left_z_mask = left_z <= torch.mean(torch.sort(left_z)[0][:, :left_z.shape[1] // 5], axis=-1)
        # left_z_mask = torch.stack([left_z_mask, left_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # left_z_mask[:, :, 1] = False  # Blank right side
        # contact_signal[left_z_mask] = 0.4
        #
        # right_z_mask = right_z <= torch.mean(torch.sort(right_z)[0][:, :right_z.shape[1] // 5], axis=-1)
        # right_z_mask = torch.stack([right_z_mask, right_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # right_z_mask[:, :, 0] = False  # Blank left side
        # contact_signal[right_z_mask] = 0.4
        # contact_signal[left_z <= (torch.mean(torch.sort(left_z)[:left_z.shape[0] // 5]) + 20), 0] = 1
        # contact_signal[right_z <= (torch.mean(torch.sort(right_z)[:right_z.shape[0] // 5]) + 20), 1] = 1

        # plt.plot(to_np_cpu(left_z[0]), label='left_z')
        # plt.plot(to_np_cpu(left_velocity[0]), label='left_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 0]), label='left_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        # plt.plot(to_np_cpu(right_z[0]), label='right_z')
        # plt.plot(to_np_cpu(right_velocity[0]), label='right_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 1]), label='right_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()

        gt_joint_xyz = gt_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]  # [BatchSize, 4, 3, Frames]
        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
        fc_mask = gt_joint_vel <= 0.01
        pred_joint_xyz = pred_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]  # [BatchSize, 4, 3, Frames]
        pred_joint_vel = torch.linalg.norm(pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
        pred_joint_vel[~fc_mask] = 0  # Blank non-contact velocities frames. [BS,4,FRAMES]
        pred_joint_vel = torch.unsqueeze(pred_joint_vel, dim=2)

        """DEBUG CODE"""
        # plt.title(f'Joint: {joint_idx}')
        # plt.plot(to_np_cpu(gt_joint_vel[0]), label='velocity')
        # plt.plot(to_np_cpu(fc_mask[0]), label='fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        return self.masked_l2(pred_joint_vel, torch.zeros(pred_joint_vel.shape, device=pred_joint_vel.device), mask[:, :, :, 1:])

    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def foot_contact_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259
        contact_mask_gt = contact > 0.5  # contact mask order for indexes are fid_l [7, 10], fid_r [8, 11]
        vel_lf_7 = local_velocity[:, 7 * 3 : 8 * 3, :, :]
        vel_rf_8 = local_velocity[:, 8 * 3 : 9 * 3, :, :]
        vel_lf_10 = local_velocity[:, 10 * 3 : 11 * 3, :, :]
        vel_rf_11 = local_velocity[:, 11 * 3 : 12 * 3, :, :]

        calc_vel_lf_7 = ric_data[:, 6 * 3 : 7 * 3, :, 1:] - ric_data[:, 6 * 3 : 7 * 3, :, :-1]
        calc_vel_rf_8 = ric_data[:, 7 * 3 : 8 * 3, :, 1:] - ric_data[:, 7 * 3 : 8 * 3, :, :-1]
        calc_vel_lf_10 = ric_data[:, 9 * 3 : 10 * 3, :, 1:] - ric_data[:, 9 * 3 : 10 * 3, :, :-1]
        calc_vel_rf_11 = ric_data[:, 10 * 3 : 11 * 3, :, 1:] - ric_data[:, 10 * 3 : 11 * 3, :, :-1]

        # vel_foots = torch.stack([vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11], dim=1)
        for chosen_vel_foot_calc, chosen_vel_foot, joint_idx, contact_mask_idx in zip(
            [calc_vel_lf_7, calc_vel_rf_8, calc_vel_lf_10, calc_vel_rf_11], [vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11], [7, 10, 8, 11], [0, 1, 2, 3]
        ):
            tmp_mask_gt = contact_mask_gt[:, contact_mask_idx, :, :].cpu().detach().numpy().reshape(-1).astype(int)
            chosen_vel_norm = np.linalg.norm(chosen_vel_foot.cpu().detach().numpy().reshape((3, -1)), axis=0)
            chosen_vel_calc_norm = np.linalg.norm(chosen_vel_foot_calc.cpu().detach().numpy().reshape((3, -1)), axis=0)

            print(tmp_mask_gt.shape)
            print(chosen_vel_foot.shape)
            print(chosen_vel_calc_norm.shape)
            import matplotlib.pyplot as plt

            plt.plot(tmp_mask_gt, label="FC mask")
            plt.plot(chosen_vel_norm, label="Vel. XYZ norm (from vector)")
            plt.plot(chosen_vel_calc_norm, label="Vel. XYZ norm (calculated diff XYZ)")

            plt.title(f"FC idx {contact_mask_idx}, Joint Index {joint_idx}")
            plt.legend()
            plt.show()
        return 0

    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def velocity_consistency_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259

        calc_vel_from_xyz = ric_data[:, :, :, 1:] - ric_data[:, :, :, :-1]
        velocity_from_vector = local_velocity[:, 3:, :, 1:]  # Slicing out root
        r_rot_quat, r_pos = motion_process.recover_root_rot_pos(target.permute(0, 2, 3, 1).type(th.FloatTensor))
        print(f"r_rot_quat: {r_rot_quat.shape}")
        print(f"calc_vel_from_xyz: {calc_vel_from_xyz.shape}")
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 2, 3, 1)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21, 3)).type(th.FloatTensor)
        r_rot_quat_adapted = r_rot_quat[..., :-1, None, :].repeat((1, 1, 1, 21, 1)).to(calc_vel_from_xyz.device)
        print(f"calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}")
        print(f"r_rot_quat_adapted: {r_rot_quat_adapted.shape}, {r_rot_quat_adapted.device}")

        calc_vel_from_xyz = motion_process.qrot(r_rot_quat_adapted, calc_vel_from_xyz)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21 * 3))
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 3, 1, 2)
        print(f"calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}")

        import matplotlib.pyplot as plt

        for i in range(21):
            plt.plot(
                np.linalg.norm(calc_vel_from_xyz[:, i * 3 : (i + 1) * 3, :, :].cpu().detach().numpy().reshape((3, -1)), axis=0), label="Calc Vel"
            )
            plt.plot(
                np.linalg.norm(velocity_from_vector[:, i * 3 : (i + 1) * 3, :, :].cpu().detach().numpy().reshape((3, -1)), axis=0), label="Vector Vel"
            )
            plt.title(f"Joint idx: {i}")
            plt.legend()
            plt.show()
        print(calc_vel_from_xyz.shape)
        print(velocity_from_vector.shape)
        diff = calc_vel_from_xyz - velocity_from_vector
        print(np.linalg.norm(diff.cpu().detach().numpy().reshape((63, -1)), axis=0))

        return 0

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise, model_kwargs=model_kwargs)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

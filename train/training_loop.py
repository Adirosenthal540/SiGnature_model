import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.beat2.networks.evaluator_wrapper import EvaluatorWrapper
from eval import base_evaluation_pipeline
from data_loaders.get_data import get_dataset_loader
from data_loaders.beat2.utils import rotation_conversions as rc

# Typing
from train.train_platforms import TrainPlatform
from model.mdm import MDM
from diffusion.respace import SpacedDiffusion
from diffusion.resample import ScheduleSampler
from torch.utils.data import DataLoader
from utils.semantic_weighting import create_semantic_weighting_from_dataset


class TrainLoop:

    def __init__(
        self,
        args,
        train_platform: TrainPlatform,
        model: MDM,
        diffusion: SpacedDiffusion,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        seg_dataset=None,
        save_dir=None,
        max_loss=None,
        min_runing_steps=1500,
    ):
        self.dataset_name = args.dataset
        assert self.dataset_name == "beat2", "Only beat2 is supported"
        self.seg_dataset = seg_dataset
        self.args = args
        self.train_platform: TrainPlatform = train_platform
        self.model: MDM = model
        self.use_ddp = False
        self.ddp_model = self.model
        self.diffusion: SpacedDiffusion = diffusion
        self.train_dataloader: DataLoader = train_dataloader
        self.val_dataloader: DataLoader = val_dataloader
        # self.eval_dataloader: DataLoader = eval_dataloader
        self.batch_size: int = args.batch_size
        self.microbatch: int = args.batch_size  # deprecating this option
        self.lr: float = args.lr
        self.log_interval: int = args.log_interval
        self.save_interval: int = args.save_interval
        self.resume_checkpoint: str = args.resume_checkpoint
        self.use_fp16: bool = False  # deprecating this option
        self.fp16_scale_growth: float = 1e-3  # deprecating this option
        self.weight_decay: float = args.weight_decay
        self.lr_anneal_steps: int = args.lr_anneal_steps

        self.step: int = 0
        self.resume_step: int = 0
        self.global_batch: int = self.batch_size  # * dist.get_world_size()
        self.num_steps: int = args.num_steps
        self.num_epochs: int = self.num_steps // len(self.train_dataloader) + 1

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir: str = args.save_dir if save_dir is None else save_dir
        self.device = args.device
        # Initialize semantic weighting if enabled
        self.semantic_weighting = None
        if hasattr(args, "use_semantic_weighting") and args.use_semantic_weighting:
            try:
                self.semantic_weighting = create_semantic_weighting_from_dataset(self.train_dataloader.dataset, device=self.device)
                # Set semantic weighting in diffusion model
                self.diffusion.set_semantic_weighting(self.semantic_weighting)
                print("✅ Semantic weighting enabled for training")
            except Exception as e:
                print(f"⚠️ Failed to initialize semantic weighting: {e}")
                self.semantic_weighting = None
        self.overwrite: bool = args.overwrite

        # Enable debug flag to plot the input batch
        self.debug: bool = False

        # Set device
        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        # Create training optimizer
        self.optimizer = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        # Create schedule sampler
        self.schedule_sampler: ScheduleSampler = create_named_schedule_sampler(name="uniform", diffusion=diffusion)

        self.max_loss = max_loss
        self.min_runing_steps = min_runing_steps
        # Create eval class and data when 'eval_during_training' is enable
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if self.dataset_name == "beat2" and args.eval_during_training:  # todo - add beat
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            self.eval_dataloader = get_dataset_loader(
                name=self.dataset_name,
                batch_size=args.batch_size,
                split=args.eval_split,
            )

            self.eval_gt_data = get_dataset_loader(
                name=self.dataset_name,
                batch_size=args.batch_size,
                split=args.eval_split,
                hml_mode="gt",
            )

            if self.dataset_name == "beat2":
                self.eval_wrapper = EvaluatorWrapper(self.dataset_name, dist_util.dev())
                self.eval_data = {
                    "test": lambda: base_evaluation_pipeline.get_mdm_loader(
                        model,
                        diffusion,
                        args.eval_batch_size,
                        self.eval_dataloader,
                        mm_num_samples,
                        mm_num_repeats,
                        self.eval_dataloader.dataset.max_length,
                        args.eval_num_samples,
                        scale=1.0,
                    )
                }
            else:
                raise ValueError(f"Unsupported dataset name [{self.dataset_name}]")

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev()))

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.optimizer.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            for motion, cond in tqdm(self.train_dataloader):
                # motion of shape [batch_size, dim_pose, 1, max_motion_length], conf is {'y': data} where data is also a dict with keys: 'mask', 'lengths', 'text', 'tokens'
                # 'mask': boolean tensor of shape [batch_size, 1, 1, max_motion_length]
                # 'lengths: int tensor of shape [batch_size] which define the length of the motion
                # 'text': list of len batch_size and define the text for each motion
                # 'tokens': list of len batch_size and define the tokens for each motion

                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond["y"] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond["y"].items()}
                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    for k, v in logger.get_current().dumpkvs().items():
                        if k == "loss":
                            print("step[{}]: loss[{:0.5f}]".format(self.step + self.resume_step, v))
                            if self.max_loss is not None and v < self.max_loss and self.step > self.min_runing_steps:
                                break
                        if k in ["step", "samples"] or "_q" in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name="Loss")

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1

            # Validation step at the end of each epoch
            if self.val_dataloader is not None:
                self.run_validation()

            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
            if self.max_loss is not None and v < self.max_loss and self.step > self.min_runing_steps:
                break

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def run_validation(self):
        self.model.eval()

        dict_sum = {}
        for motion, cond in tqdm(self.val_dataloader):
            motion = motion.to(self.device)
            cond["y"] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond["y"].items()}

            dict_loss = self.run_eval(motion, cond)
            for k, v in dict_loss.items():
                if k not in dict_sum:
                    dict_sum[k] = v
                else:
                    dict_sum[k] += v

        for k, v in dict_sum.items():
            if k == "loss":
                print("validation step[{}]: loss[{:0.5f}]".format(self.step + self.resume_step, v / len(self.val_dataloader)))

            if k in ["step", "samples"] or "_q" in k:
                continue
            else:
                self.train_platform.report_scalar(
                    name=k + "_val",
                    value=v / len(self.val_dataloader),
                    iteration=self.step,
                    group_name="Loss val",
                )

        self.model.train()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.dataset_name == "beat2":
            dict_sum = {}
            for motion, cond in tqdm(self.eval_dataloader):
                motion = motion.to(self.device)
                cond["y"] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond["y"].items()}

                dict_loss = self.run_eval(motion, cond)
                for k, v in dict_loss.items():
                    if k not in dict_sum:
                        dict_sum[k] = v
                    else:
                        dict_sum[k] += v

            for k, v in dict_sum.items():
                if k == "loss":
                    print("evaluation step[{}]: loss[{:0.5f}]".format(self.step + self.resume_step, v / len(self.eval_dataloader)))

                if k in ["step", "samples"] or "_q" in k:
                    continue
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=v / len(self.eval_dataloader),
                        iteration=self.step,
                        group_name="Loss",
                    )

        end_eval = time.time()
        print(f"Evaluation time: {round(end_eval-start_eval)/60}min")
        # return
        if self.eval_wrapper is not None:
            print("Running evaluation loop: [Should take about 90 min]")
            log_file = os.path.join(self.save_dir, f"eval_humanml_{(self.step + self.resume_step):09d}.log")
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = base_evaluation_pipeline.evaluation(
                self.eval_wrapper,
                self.eval_gt_data,
                self.eval_data,
                log_file,
                replication_times=self.args.eval_rep_times,
                diversity_times=diversity_times,
                mm_num_times=mm_num_times,
                run_mm=False,
            )

            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith("R_precision"):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(
                            name=f"top{i + 1}_" + k,
                            value=v[i],
                            iteration=self.step + self.resume_step,
                            group_name="Eval",
                        )
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=v,
                        iteration=self.step + self.resume_step,
                        group_name="Eval",
                    )

        elif self.dataset_name in ["humanact12", "uestc"]:
            eval_args = SimpleNamespace(
                num_seeds=self.args.eval_rep_times,
                num_samples=self.args.eval_num_samples,
                batch_size=self.args.eval_batch_size,
                device=self.device,
                guidance_param=1,
                dataset=self.dataset_name,
                unconstrained=self.args.unconstrained,
                model_path=os.path.join(self.save_dir, self.ckpt_file_name()),
            )
            eval_dict = eval_humanact12_uestc.evaluate(
                eval_args,
                model=self.model,
                diffusion=self.diffusion,
                data=self.train_dataloader.dataset,
            )
            print(f'Evaluation results on {self.dataset_name}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if "unconstrained" not in k:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name="Eval",
                    )
                else:
                    self.train_platform.report_scalar(
                        name=k,
                        value=np.array(v).astype(float).mean(),
                        iteration=self.step,
                        group_name="Eval Unconstrained",
                    )

        end_eval = time.time()
        print(f"Evaluation time: {round(end_eval-start_eval)/60}min")

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.optimizer)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            if not isinstance(self.train_dataloader, list):
                dataset = self.train_dataloader.dataset
            elif self.seg_dataset is not None:
                dataset = self.seg_dataset
            else:
                dataset = None

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=dataset,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)

    def run_eval(self, batch, cond):
        return self.forward_eval(batch, cond)
        # self.log_step()

    def forward_eval(self, batch, cond):
        with torch.no_grad():
            for i in range(0, batch.shape[0], self.microbatch):
                # Eliminates the microbatch feature
                assert i == 0
                assert self.microbatch == self.batch_size
                micro = batch
                micro_cond = cond
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,  # [bs, ch, image_size, image_size]
                    t,  # [bs](int) sampled timesteps
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                # if isinstance(self.schedule_sampler, LossAwareSampler):
                #     self.schedule_sampler.update_with_local_losses(
                #         t, losses["loss"].detach()
                #     )

                # loss = (losses["loss"] * weights).mean()
                # logger.info(f"# evaluation losses #")
                dict_loss = loss_dict(
                    self.diffusion,
                    t,
                    {k: v * weights for k, v in losses.items()},
                )
        return dict_loss

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if "clip_model." in e]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.optimizer.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def loss_dict(diffusion, ts, losses):
    dict_res = {}
    for key, values in losses.items():
        dict_res[key] = values.mean().item()
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            dict_res[f"{key}_q{quartile}"] = sub_loss
    return dict_res

# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import shutil
import json

import torch
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from train.train_platforms import get_train_platform
from data_loaders.beat2.utils.build_vocab import Vocab
from data_loaders.beat2.utils.cache_utils import calculate_mean_std
import re


def main():
    args = train_args()

    train_platform = get_train_platform(train_platform_type=args.train_platform_type, save_dir=args.save_dir, args=args)

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    emage_path = os.path.join(args.save_dir, "emage.yaml")
    shutil.copy2("./dataset/emage.yaml", emage_path)

    workspace_path = os.path.join(args.save_dir, "launch.json")
    workspace_source = "./signature.code-workspace" if os.path.exists("./signature.code-workspace") else "./signature-template.code-workspace"
    shutil.copy2(workspace_source, workspace_path)

    dist_util.setup_dist(args.device)

    print("creating training data loader...")
    train_dataloader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
        split="train",
        use_amass=args.use_amass,
        use_seg=args.use_seg,
    )

    print("creating validation data loader...")
    val_dataloader = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size, split="val", device=args.device, use_amass=args.use_amass, use_seg=args.use_seg
    )
    if args.use_amass or args.use_seg:
        index_person = os.path.basename(train_dataloader.dataset.datasets[0].args.cache_path).split("_")[-1]
        args.data_path = train_dataloader.dataset.datasets[0].args.data_path
    else:
        index_person = os.path.basename(train_dataloader.dataset.args.cache_path).split("_")[-1]
        args.data_path = train_dataloader.dataset.args.data_path

    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, do_not_use_clip=args.do_not_use_clip)

    if args.model_path is not None:
        model_ckp_path = args.model_path
        print(f"Loading checkpoints from [{model_ckp_path}]...")
        state_dict = torch.load(model_ckp_path, map_location=torch.device(args.device))
        load_model_wo_clip(model, state_dict)

    assert f"_{index_person}_" in args.save_dir.split("/")[-1], "The model and the dataset doesnt match"

    model.to(args.device)

    print("Total params: %.2fM" % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")

    TrainLoop(args, train_platform, model, diffusion, train_dataloader, val_dataloader).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()

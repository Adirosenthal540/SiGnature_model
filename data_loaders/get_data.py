import torch
from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, beat2_collate


def get_dataset_class(name):
    if name == "beat2":
        from .beat2.beat2_dataset import BEAT2Dataset

        return BEAT2Dataset
    elif name == "seg":
        from .seg.seg_dataset import SegDataset

        return SegDataset
    elif name == "amass":
        from .beat2.amass_dataset import AMASSDataset

        return AMASSDataset
    else:
        raise ValueError(f"Unsupported dataset name [{name}]")


def get_collate_fn(name):
    if name == "beat2":
        return beat2_collate
    else:
        return all_collate


def get_dataset(name, split="train", device=0, dataset_cache_path=None, build_cache=True):
    DATA = get_dataset_class(name)
    if name == "beat2":
        dataset = DATA(split=split, device=device, cache_path=dataset_cache_path, build_cache=build_cache)
    elif name == "amass":
        dataset = DATA(split=split, device=device)
    elif name == "seg":
        dataset = DATA(split=split, device=device)
    else:
        raise ValueError(f"Unsupported dataset name [{name}]")

    return dataset


def get_dataset_loader(name, batch_size, split: str = "train", device="cpu", shuffle=True, use_amass=False, use_seg=False, dataset_cache_path=None, build_cache=True):
    dataset = get_dataset(name, split, device, dataset_cache_path, build_cache=build_cache)
    collate = get_collate_fn(name)

    if use_amass:
        # train_data_amass = __import__(f"dataloaders.amass_sep_lower_h3d", fromlist=["something"]).CustomDataset(args, "train")
        dataset_amass = get_dataset("amass", split, device)
        if not (use_seg or split != "train"):
            dataset = torch.utils.data.ConcatDataset(
                [
                    # dataset,
                    dataset_amass,
                ]
            )
    if use_seg and split == "train":
        # train_data_amass = __import__(f"dataloaders.amass_sep_lower_h3d", fromlist=["something"]).CustomDataset(args, "train")
        dataset_seg = get_dataset("seg", split, device)
        if use_amass:
            dataset = torch.utils.data.ConcatDataset(
                [
                    dataset,
                    dataset_amass,
                    dataset_seg,
                ]
            )
        else:
            dataset = torch.utils.data.ConcatDataset(
                [
                    dataset,
                    dataset_seg,
                ]
            )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # insert as input
        drop_last=True,
        collate_fn=collate,
    )

    return loader

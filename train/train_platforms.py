import os
from abc import ABC, abstractmethod
from typing import Dict


class TrainPlatform(ABC):
    def __init__(self, save_dir: str, args: Dict):
        raise NotImplementedError

    @abstractmethod
    def report_scalar(self, name, value, iteration, group_name=None):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir: str, args: Dict):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir: str, args: Dict):
        from clearml import Task

        path, name = os.path.split(save_dir)
        self.task = Task.init(
            project_name="motion_diffusion", task_name=name, output_uri=path
        )
        self.task.connect(args, name="Args")
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(
            title=group_name, series=name, iteration=iteration, value=value
        )

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir: str, args: Dict):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f"{group_name}/{name}", value, iteration)

    def close(self):
        self.writer.close()


class WandbPlatform(TrainPlatform):
    def __init__(self, save_dir: str, args: Dict):

        # Add your own values
        self.entity: str = args.entity
        self.project: str = args.project

        import wandb

        self.runwandb = wandb.init(
            reinit=True, config=args, entity=self.entity, project=self.project
        )

    def report_scalar(self, name, value, iteration, group_name=None):
        log_wandb = {name: value}
        self.runwandb.log(log_wandb, step=iteration)

    def close(self):
        self.runwandb.finish()


def get_train_platform(
    train_platform_type: str, save_dir: str, args: Dict
) -> TrainPlatform:

    platforms = {
        "NoPlatform": NoPlatform,
        "ClearmlPlatform": ClearmlPlatform,
        "TensorboardPlatform": TensorboardPlatform,
        "WandbPlatform": WandbPlatform,
    }
    platform_class = platforms.get(train_platform_type)
    if platform_class is None:
        raise ValueError(f"Platform type '{train_platform_type}' is not supported.")

    return platform_class(save_dir, args)

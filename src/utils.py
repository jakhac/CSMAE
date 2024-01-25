import json
import os
import random
import string
import time
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback

class Checkpointer(Callback):
    def __init__(
        self,
        cfg: DictConfig,
        logdir: Union[str, Path] = Path("trained_models"),
    ):
        """Custom checkpointer callback that stores checkpoints in an easier to access way.

        Args:
            cfg (DictConfig): DictConfig containing at least an attribute name.
            logdir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
        """

        super().__init__()

        self.cfg = cfg
        self.logdir = Path(logdir)

        self.ckpt_placeholder = None
        self.path = ""

    @staticmethod
    def random_string(letter_count=4, digit_count=4):
        tmp_random = random.Random(time.time())
        rand_str = "".join((tmp_random.choice(string.ascii_lowercase) for _ in range(letter_count)))
        rand_str += "".join((tmp_random.choice(string.digits) for _ in range(digit_count)))
        rand_str = list(rand_str)
        tmp_random.shuffle(rand_str)
        return "".join(rand_str)

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.logger is None:
            if self.logdir.exists():
                existing_versions = set(os.listdir(self.logdir))
            else:
                existing_versions = []
            version = "offline-" + self.random_string()
            while version in existing_versions:
                version = "offline-" + self.random_string()
        else:
            version = str(trainer.logger.version)
            self.wandb_run_id = version

        if version is not None:
            self.path = self.logdir / version
            self.ckpt_placeholder = f"{self.cfg.name}-{version}" + "-ep={}.ckpt"
        else:
            self.path = self.logdir
            self.ckpt_placeholder = f"{self.cfg.name}" + "-ep={}.ckpt"
        self.last_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def save_args(self, trainer: pl.Trainer):
        """Stores arguments into a json file.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.is_global_zero:
            yaml_path = self.path / "args.yaml"
            OmegaConf.save(self.cfg, yaml_path)

    def save(self, trainer: pl.Trainer):
        """Saves current checkpoint.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.is_global_zero and not trainer.sanity_checking:
            epoch = trainer.current_epoch  # type: ignore
            ckpt = self.path / self.ckpt_placeholder.format(epoch)
            trainer.save_checkpoint(ckpt)

            if self.last_ckpt and self.last_ckpt != ckpt:
                os.remove(self.last_ckpt)
            self.last_ckpt = ckpt

    def on_train_start(self, trainer: pl.Trainer, _):
        """Executes initial setup and saves arguments.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer)
        self.save_args(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """Tries to save current checkpoint at the end of each train epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = trainer.current_epoch
        if epoch % 1 == 0:
            self.save(trainer)


def _indent(s, num_spaces=0, indent_first=False):
    """
    Indents a string by a number of spaces. Indents every line individually except for
    the first line if indent_first is not set
    :param s: string to indent
    :param num_spaces: number of spaces to indent
    :param indent_first: if set, first line is indented as well, otherwise no spaces
                         added to first line
    :return: s with each line indented
    """
    # split by newline
    s = str.split(s, "\n")
    # add num_spaces spaces to front of each line except the first
    if indent_first:
        s = [(num_spaces * " ") + str.lstrip(line) for line in s]
    else:
        first_line = s[0]
        s = [(num_spaces * " ") + str.lstrip(line) for line in s[1:]]
        s = [first_line] + s
    # join with spaces
    s = "\n".join(s)
    return s


class Messages:
    # color prefixes for use in print
    # e.g. print(f"{bcolors.WARNING}This is a Warning.{bcolors.ENDC}")
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def warn(cls, message: str):
        print(f"\n{cls.WARNING}WARNING: {_indent(message, 9)}{cls.ENDC}")

    @classmethod
    def error(cls, message: str):
        print(f"\n{cls.FAIL}ERROR: {_indent(message, 7)}{cls.ENDC}")

    @classmethod
    def success(cls, message: str):
        print(f"\n{cls.OKGREEN}SUCCESS: {_indent(message, 10)}{cls.ENDC}")

    @classmethod
    def hint(cls, message: str):
        print(f"\n{cls.OKCYAN}HINT: {_indent(message, 6)}{cls.ENDC}")


def omegaconf_select(cfg, key, default=None):
    """Wrapper for OmegaConf.select to allow None to be returned instead of 'None'."""
    value = OmegaConf.select(cfg, key, default=default)
    if value == "None":
        return None
    return value
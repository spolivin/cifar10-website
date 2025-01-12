"""Module for Trainer class."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from typing import Any, TypeAlias

import torch
from tqdm import tqdm

from .checkpoint_handler import CheckpointHandler
from .metrics_tracker import MetricsTracker

TorchDataloader: TypeAlias = torch.utils.data.dataloader.DataLoader


class Trainer:
    """jjjj"""

    def __init__(
        self,
        model: Any,
        criterion: Any,
        optimizer: Any,
        train_loader: TorchDataloader,
        valid_loader: TorchDataloader,
        train_on: str = "auto",
    ) -> None:
        """_summary_

        Args:
            model (Any): _description_
            criterion (Any): _description_
            optimizer (Any): _description_
            train_loader (TorchDataloader): _description_
            valid_loader (TorchDataloader): _description_
            callbacks (List[Any], optional): _description_. Defaults to [].
            train_on (str, optional): _description_. Defaults to "auto".
            config (_type_, optional): _description_. Defaults to { "log_to_file": False, "log_to_tensorboard": False, "log_to_sqlite": False, }.
        """
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if train_on == "auto"
            else torch.device(train_on)
        )
        self.model = model.to(self.device)
        self.optimizer = type(optimizer)(
            self.model.parameters(), **optimizer.defaults
        )
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.scheduler = None
        self.scheduler_level = None

        self.last_epoch = 0

        self.metrics_tracker = MetricsTracker()
        self.ckpt_handler = CheckpointHandler()

    def set_scheduler(
        self,
        scheduler_class: Any,
        scheduler_params: dict[str, Any],
        level="epoch",
    ) -> None:
        if level not in ("epoch", "batch"):
            raise ValueError("Invalid value")
        else:
            self.scheduler_level = level
        self.scheduler = scheduler_class(self.optimizer, **scheduler_params)

    def reset_scheduler(self) -> None:
        self.scheduler = None
        self.scheduler_level = None

    def __call__(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            x_batch (torch.Tensor): _description_
            y_batch (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        outputs = self.model(x_batch)
        loss = self.criterion(outputs, y_batch.long())

        return loss, outputs

    def run(
        self,
        epochs: int,
        seed: int = 42,
        enable_tqdm: bool = True,
    ) -> None:
        """_summary_

        Args:
            epochs (int): _description_
            run_id (Optional[str], optional): _description_. Defaults to None.
            seed (int, optional): _description_. Defaults to 42.
            enable_tqdm (bool, optional): _description_. Defaults to True.
        """

        total_epochs = epochs + self.last_epoch
        for epoch in range(self.last_epoch, total_epochs):
            self.train_loop(
                epoch=epoch, epochs=total_epochs, enable_tqdm=enable_tqdm
            )
            self.validate_loop(
                epoch=epoch, epochs=total_epochs, enable_tqdm=enable_tqdm
            )
            self.last_epoch = epoch + 1
            self.metrics_tracker.reset()

    def train_loop(self, epoch, epochs: int, enable_tqdm: bool) -> None:
        """_summary_

        Args:
            epoch (int): _description_
        """

        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}/{epochs} [Training]",
            position=0,
            leave=True,
            unit="batches",
            disable=not enable_tqdm,
        ) as pbar:
            pbar.set_postfix({"lr": self.optimizer.param_groups[0]["lr"]})
            self.model.train()
            for batch in self.train_loader:
                self.process_batch(batch=batch, split="train")
                pbar.update(1)
            if self.scheduler and self.scheduler_level == "epoch":
                self.scheduler.step()

    def validate_loop(self, epoch, epochs, enable_tqdm):
        with tqdm(
            total=len(self.valid_loader),
            desc=f"Epoch {epoch + 1}/{epochs} [Validation]",
            position=0,
            leave=True,
            unit="batches",
            disable=not enable_tqdm,
        ) as pbar:
            self.model.eval()
            with torch.no_grad():
                for batch in self.valid_loader:
                    self.process_batch(batch=batch, split="valid")
                    pbar.update(1)

            self.metrics = self.metrics_tracker.get_all_metrics()
            pbar.set_postfix(
                {
                    "loss": (
                        round(self.metrics["loss/train"], 4),
                        round(self.metrics["loss/valid"], 4),
                    ),
                    "acc": (
                        round(self.metrics["accuracy/train"], 4),
                        round(self.metrics["accuracy/valid"], 4),
                    ),
                }
            )

    def process_batch(self, batch: list[torch.Tensor], split: str) -> None:
        """_summary_

        Args:
            batch (List[torch.Tensor]): _description_
            split (str): _description_
        """

        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        loss, outputs = self(x_batch=x_batch, y_batch=y_batch)

        if split == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler and self.scheduler_level == "batch":
                self.scheduler.step()

        self.collect_metrics(
            loss=loss, outputs=outputs, labels=y_batch, split=split
        )

    def collect_metrics(
        self,
        loss: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        split: str,
    ) -> None:
        """_summary_

        Args:
            loss (torch.Tensor): _description_
            outputs (torch.Tensor): _description_
            labels (torch.Tensor): _description_
            split (str): _description_
        """

        loss = loss.item()
        self.metrics_tracker.update(f"loss/{split}", loss)

        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == labels).sum().cpu().item()
        total = labels.size(0)
        self.metrics_tracker.update_accuracy(correct, total, split=split)

    def save_checkpoint(self, path: str) -> None:
        """_summary_

        Args:
            path (str): _description_
        """
        self.ckpt_handler.save_checkpoint(path=path, trainer=self)

    def load_checkpoint(self, path: str) -> None:
        """_summary_

        Args:
            path (str): _description_
        """
        self.ckpt_handler.load_checkpoint(path=path, trainer=self)

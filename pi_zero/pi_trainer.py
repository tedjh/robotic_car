from pathlib import Path

import mlflow
import torch
from pi_zero.pi_model import SmallPi0
from torch import nn
from tqdm import tqdm

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


class PiTrainer:
    def __init__(
        self,
        pi: SmallPi0,
        training_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        experiment_name: str = "small_pi0",
        run_name: str | None = None,
        checkpoint_dir: Path | None = None,
        checkpoint_freq: int = 10,
    ):
        self.pi: SmallPi0 = pi
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.pi.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.epochs = epochs
        self.training_loader = training_loader
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq

    def train_step(
        self,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        image: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:
        self.optimizer.zero_grad()
        device = self.pi.device
        batch_size, action_horizon, action_dim = actions.shape

        actions = actions.to(device)
        noised_actions = torch.randn(
            batch_size, action_horizon, action_dim, device=self.pi.device
        )
        noise_level = torch.rand(batch_size, device=self.pi.device)

        interpolated_actions = (
            1 - noise_level[:, None, None]
        ) * noised_actions + noise_level[:, None, None] * actions
        target_vector_field = actions - noised_actions

        pred: torch.Tensor = self.pi(
            image.to(device),
            prompt_tokens.to(device),
            prompt_mask.to(device),
            state.to(device),
            interpolated_actions.to(device),
            noise_level.to(device),
        )
        loss = nn.functional.mse_loss(pred[:, 1:, :], target_vector_field)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _log_params(self) -> None:
        trainable = sum(p.numel() for p in self.pi.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.pi.parameters())
        mlflow.log_params(
            {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
                "batch_size": self.training_loader.batch_size,
                "n_action_layers": self.pi.n_action_layers,
                "n_action_attention_heads": self.pi.n_action_attention_heads,
                "gemma_hidden_dim": self.pi.gemma_hidden_dim,
                "trainable_params": trainable,
                "total_params": total,
            }
        )

    def _save_checkpoint(self, epoch: int) -> None:
        assert self.checkpoint_dir is not None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.pi.trainable_state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        mlflow.log_artifact(str(path))

    def run(self) -> None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            self._log_params()
            global_step = 0
            pbar = tqdm(range(self.epochs))
            for epoch in pbar:
                epoch_losses = []
                for batch in self.training_loader:
                    loss = self.train_step(**batch)
                    mlflow.log_metric("train/loss", loss, step=global_step)
                    epoch_losses.append(loss)
                    global_step += 1
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                mlflow.log_metric("train/loss_epoch_mean", mean_loss, step=epoch)
                pbar.set_postfix(loss=epoch_losses[-1], epoch=epoch)
                if (
                    self.checkpoint_dir is not None
                    and (epoch + 1) % self.checkpoint_freq == 0
                ):
                    self._save_checkpoint(epoch)
        if self.checkpoint_dir is not None and self.epochs % self.checkpoint_freq != 0:
            self._save_checkpoint(self.epochs - 1)

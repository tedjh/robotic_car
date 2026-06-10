from pathlib import Path

import torch
from pi_zero.pi_model import SmallPi0
from pi_zero.pi_trainer import PiTrainer
from torch.utils.data import DataLoader


def main(
    training_data_path: Path,
    model_id: str = "google/paligemma2-3b-pt-224",
    batch_size: int = 32,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(training_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batch = next(iter(dataloader))
    state_dim, action_dim = batch["state"].shape[-1], batch["actions"].shape[-1]
    pi = SmallPi0.from_pretrained(
        model_id, device, action_dim=action_dim, state_dim=state_dim
    )
    print("Model loaded successfully.")

    trainer = PiTrainer(pi, training_loader=dataloader, epochs=10)
    print("Training starting...")
    trainer.run()


if __name__ == "__main__":
    training_data_path = Path("robotic_car") / "training_data" / "driving_dataset.pt"
    main(training_data_path)

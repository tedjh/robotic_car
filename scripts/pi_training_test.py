from pathlib import Path

import torch
from pi_zero.pi_model import SmallPi0
from pi_zero.pi_trainer import PiTrainer
from transformers import AutoTokenizer

ROOT = Path(__file__).parents[1]


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main(
    prompts: list[str] = ["drive forward", "go to the living room", "brake"],
    action_horizon: int = 10,
    action_dim: int = 2,
    model_id: str = "google/paligemma2-3b-pt-224",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = len(prompts)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    state_dim = action_dim  # Assume state_dim is the same as action_dim in this example

    pi = SmallPi0.from_pretrained(
        pretrained_model_name_or_path=model_id,
        device=device,
        action_dim=action_dim,
        state_dim=state_dim,
        cache_dir=ROOT / "models",
    )
    print("Model loaded successfully.")

    images = torch.randn(batch_size, 3, 224, 224, device=device)
    tokenized_prompts = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )
    prompt_tokens = tokenized_prompts.input_ids.to(device)
    prompt_mask = tokenized_prompts.attention_mask.to(device)
    state = torch.randn(batch_size, action_dim, device=device)
    actions = torch.randn(batch_size, action_horizon, action_dim, device=device)

    training_loader = torch.utils.data.DataLoader(
        ListDataset(
            [
                {
                    "image": images[i],
                    "prompt_tokens": prompt_tokens[i],
                    "prompt_mask": prompt_mask[i],
                    "state": state[i],
                    "actions": actions[i],
                }
                for i in range(batch_size)
            ]
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    print("Dataloader built")

    trainer = PiTrainer(pi, training_loader=training_loader, epochs=1)

    print("Training starting...")
    trainer.run()


if __name__ == "__main__":
    main()

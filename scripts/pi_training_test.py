import torch
from pi_zero.pi_model import SmallPi0
from pi_zero.pi_trainer import PiTrainer
from transformers import AutoTokenizer


def main(
    prompts: list[str] = ["drive forward", "go to the living room", "brake"],
    action_horizon: int = 10,
    action_dim: int = 2,
    model_id: str = "google/paligemma2-3b-pt-224",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = len(prompts)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pi = SmallPi0.from_pretrained(model_id, device, action_dim=action_dim)
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
    noised_actions = torch.randn(batch_size, action_horizon, action_dim, device=device)
    noise_level = torch.rand(batch_size, device=device)
    target_actions = torch.randn(batch_size, action_horizon, action_dim, device=device)

    training_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            images,
            prompt_tokens,
            prompt_mask,
            state,
            noised_actions,
            noise_level,
            target_actions,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    print("Dataloader built")

    trainer = PiTrainer(
        pi, training_loader=training_loader
    )  # Assuming you have a training loader
    print("Training starting...")
    trainer.run()


if __name__ == "__main__":
    main()

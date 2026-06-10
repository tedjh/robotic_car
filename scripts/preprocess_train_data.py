import csv
from pathlib import Path

import torch
from torchvision.io import read_image
from transformers import AutoTokenizer, SiglipProcessor

example_data_path = Path(r"\\wsl.localhost\Ubuntu\home\tedjh\robotic_car\training_data")
out_data = Path("training_data")
speed_max = 255


def preprocess_state(state_str: str) -> torch.Tensor:
    # state_str has form "b'0,1,0,1\n'".
    state_str = state_str.strip().strip("b'")[:-2]
    left_speed, left_dir, right_speed, right_dir = map(float, state_str.split(","))
    left_speed, right_speed = left_speed / speed_max, right_speed / speed_max
    if left_dir == 0:
        left_speed *= -1
    if right_dir == 0:
        right_speed *= -1
    return torch.tensor([left_speed, right_speed])


def main(
    action_horizon: int = 10,
    model_id: str = "google/paligemma2-3b-pt-224",
) -> None:
    dataset = []
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    for episode_dir in example_data_path.iterdir():
        if not episode_dir.is_dir():
            continue

        print(f"Processing {episode_dir.name}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        with open(episode_dir / "labels.csv", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))  # header consumed automatically

        states_list = [preprocess_state(row["state"]) for row in rows]
        tokenized_prompt = None
        for i, row in enumerate(rows):
            if (
                i > len(rows) - action_horizon - 1
            ):  # Skip last few rows without enough future actions
                break

            if i == 0:
                tokenized_prompt = tokenizer(
                    row["input_prompt"],
                    padding=True,
                    return_tensors="pt",
                )
            image_tensor = read_image(str(episode_dir / row["image_filename"]))
            image_tensor = processor(image_tensor, return_tensors="pt")["pixel_values"]  # type: ignore
            assert tokenized_prompt is not None, (
                "Tokenized prompt should have been set by now"
            )
            dataset.append(
                {
                    "prompt_tokens": tokenized_prompt.input_ids.squeeze(0),
                    "prompt_mask": tokenized_prompt.attention_mask.squeeze(0),
                    "image": image_tensor.squeeze(0),  # e.g. shape [3, 224, 224]
                    "state": states_list[i],  # e.g. shape [2] for linear/angular
                    "actions": torch.stack(
                        states_list[i + 1 : i + action_horizon + 1]
                    ),  # shape [action_horizon, 2]
                }
            )

        print(f"Finished processing {episode_dir.name}")

    torch.save(dataset, out_data / "driving_dataset.pt")


if __name__ == "__main__":
    main()

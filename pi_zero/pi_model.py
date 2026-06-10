import math
from pathlib import Path

import mlflow
import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Gemma2Model,
    PaliGemmaForConditionalGeneration,
)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def sinusoidal_positional_encoding(
    t: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """Encode a (batch of) flow matching timestep(s) as a `dim`-vector.

    Args:
        t: Timesteps of shape (batch_size,) or scalar. Typically in [0, 1].
        dim: Output embedding dimension. Must be even.
        max_period: Controls the minimum frequency of the encoding.

    Returns:
        Tensor of shape (batch_size, dim).
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    t = torch.atleast_1d(t).to(dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ActionExpertLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_dim // num_attention_heads
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.q_proj = nn.Linear(
            self.hidden_dim,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_dim,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_dim,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_dim,
            bias=False,
        )
        self.layer_norm_1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attn_keys: torch.Tensor,
        cross_attn_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        # residual: (batch, action_horizon_length + 1, gemma_hidden_dim)
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)

        input_shape = hidden_states.shape[:-1]

        # hidden_shape: (batch, action_horizon_length + 1, num_attention_heads, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # hidden_states: (batch, action_horizon_length + 1, gemma_hidden_dim)
        # -> (batch, action_horizon_length + 1, num_attention_heads * head_dim)
        # -> (batch, action_horizon_length + 1, num_attention_heads, head_dim)
        # -> query_states: (batch, num_attention_heads, action_horizon_length + 1, head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # cross_attn_keys: (batch, num_attention_heads, image_tokens + prompt_length, head_dim)
        # key_states: (batch, num_attention_heads, action_horizon_length + 1 + image_tokens + prompt_length, head_dim)
        key_states = torch.cat([key_states, cross_attn_keys], dim=2)
        value_states = torch.cat([value_states, cross_attn_values], dim=2)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Apply attention mask. First broadcast, then convert to additive mask.
        # mask: (batch, 1, 1, action_horizon_length + 1 + image_tokens + prompt_length)
        mask = (1.0 - attention_mask[:, None, None, :].to(attn_weights.dtype)) * -1e9
        attn_weights = attn_weights + mask

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        # attn_output: (batch, num_attention_heads, action_horizon_length + 1, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)
        # attn_output: (batch, action_horizon_length + 1, num_attention_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output: (batch, action_horizon_length + 1, num_attention_heads * head_dim)
        attn_output = attn_output.view(*input_shape, self.hidden_dim)
        # hidden_states: (batch, action_horizon_length + 1, gemma_hidden_dim)
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SmallPi0(nn.Module):
    def __init__(
        self,
        full_model: PaliGemmaForConditionalGeneration,
        state_dim: int = 2,
        action_dim: int = 2,
    ):
        super().__init__()

        self.vision_encoder = full_model.model.vision_tower  # SiglipVisionModel
        self.projector = full_model.model.multi_modal_projector  # linear projection
        assert isinstance(full_model.model.language_model, Gemma2Model)
        self.gemma: Gemma2Model = full_model.model.language_model
        assert isinstance(self.gemma.layers, nn.ModuleList)
        self.n_action_layers = self.gemma.config.num_hidden_layers
        self.n_action_attention_heads = self.gemma.config.num_key_value_heads

        # Freeze all PaliGemma weights
        for p in [self.vision_encoder, self.projector, self.gemma]:
            p.requires_grad_(False)

        action_gemma_hidden_dim = (
            self.n_action_attention_heads * self.gemma.config.head_dim
        )
        # Trainable action expert components
        self.state_embedding = nn.Linear(state_dim, action_gemma_hidden_dim)
        self.action_embedding_1 = nn.Linear(action_dim, action_gemma_hidden_dim)
        self.action_embedding_2 = nn.Sequential(
            nn.Linear(action_gemma_hidden_dim * 2, action_gemma_hidden_dim),
            nn.SiLU(),
            nn.Linear(action_gemma_hidden_dim, action_gemma_hidden_dim),
        )

        self.action_expert_layers = nn.ModuleList(
            [
                ActionExpertLayer(
                    action_gemma_hidden_dim,
                    num_attention_heads=self.n_action_attention_heads,
                )
                for _ in range(self.n_action_layers)
            ]
        )
        self.action_head = nn.Linear(action_gemma_hidden_dim, action_dim)
        self.gemma_hidden_dim = action_gemma_hidden_dim

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        state: torch.Tensor,
        noised_actions: torch.Tensor,
        noise_level: torch.Tensor,
    ):
        """Forward pass for SmallPi0.

        This predicts the vector field over action tokens.
        """
        if not (
            isinstance(self.vision_encoder, nn.Module)
            and isinstance(self.projector, nn.Module)
            and isinstance(self.gemma, nn.Module)
        ):
            raise ValueError("PaliGemma components must be nn.Modules")

        if not isinstance(self.gemma.embed_tokens, nn.Embedding):
            raise ValueError("PaliGemma's embed_tokens must be an nn.Embedding")

        # image_embeds: (batch, num_image_tokens, gemma_hidden_dim)
        image_embeds: torch.Tensor = self.projector(
            self.vision_encoder(image).last_hidden_state
        )

        # prompt_tokens: (batch, prompt_length,) --> (batch, prompt_length, gemma_hidden_dim)
        prompt_embeds: torch.Tensor = self.gemma.embed_tokens(prompt_tokens)

        image_mask = torch.ones(
            image_embeds.shape[0],  # batch size
            image_embeds.shape[1],  # num_image_tokens
            device=image.device,
            dtype=prompt_mask.dtype,
        )
        # full_mask: (batch, num_image_tokens + prompt_length)
        full_mask = torch.cat([image_mask, prompt_mask], dim=1)

        context = self.gemma(
            inputs_embeds=torch.concat([image_embeds, prompt_embeds], dim=1),
            use_cache=True,
            attention_mask=full_mask,
        )

        # state_embeds: (batch, state_dim) --> (batch, gemma_hidden_dim)
        state_embeds: torch.Tensor = self.state_embedding(state)

        # action_embeds: (batch, action_horizon_length, action_dim)
        # --> (batch, action_horizon_length, gemma_hidden_dim)
        # --> (batch, action_horizon_length, gemma_hidden_dim * 2)  concat pos encoding
        # --> (batch, action_horizon_length, gemma_hidden_dim)
        action_embeds: torch.Tensor = self.action_embedding_1(noised_actions)
        # pos_embeds: (batch, gemma_hidden_dim)
        pos_embeds = sinusoidal_positional_encoding(noise_level, self.gemma_hidden_dim)
        action_embeds: torch.Tensor = torch.concat(
            [
                action_embeds,
                pos_embeds.unsqueeze(1).expand(-1, action_embeds.shape[1], -1),
            ],
            dim=-1,
        )
        action_embeds = self.action_embedding_2(action_embeds)

        # state_action_embeds: (batch, action_horizon_length + 1, gemma_hidden_dim)
        state_action_embeds = torch.concat(
            [state_embeds.unsqueeze(1), action_embeds], dim=1
        )

        # expert_mask: (batch, action_horizon_length + 1) - all ones since the action
        # expert will attend to all state and action tokens.
        expert_mask = torch.ones(
            state_action_embeds.shape[0],
            state_action_embeds.shape[1],
            device=image.device,
            dtype=full_mask.dtype,
        )
        # cross_mask: (batch, action_horizon_length + 1 + num_image_tokens + prompt_length)
        cross_mask = torch.cat([expert_mask, full_mask], dim=1)

        for layer_idx in range(self.n_action_layers):
            state_action_embeds = self.action_expert_layers[layer_idx](
                state_action_embeds,
                # cross_attn_keys: (batch, num_kv_heads, seq_len, head_dim)
                cross_attn_keys=context.past_key_values.layers[layer_idx].keys,
                cross_attn_values=context.past_key_values.layers[layer_idx].values,
                attention_mask=cross_mask,
            )

        return self.action_head(state_action_embeds)  # predicted velocity field

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "google/paligemma2-3b-pt-224",
        device: torch.device | None = None,
        **kwargs,
    ):
        """Load SmallPi0 from a pretrained PaliGemma checkpoint."""
        full_model = PaliGemmaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=Path(__file__).parent.parent / "models" / "paligemma2-3b-pt-224",
            torch_dtype=torch.bfloat16,
            device_map=device if device is not None else "auto",
            local_files_only=True,
        )
        return cls(full_model=full_model, **kwargs).to(device)


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

    def train_step(
        self,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        state: torch.Tensor,
        noised_actions: torch.Tensor,
        noise_level: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> float:
        self.optimizer.zero_grad()
        device = self.pi.device
        interpolated_actions = (
            1 - noise_level[:, None, None]
        ) * noised_actions + noise_level[:, None, None] * target_actions
        target_vector_field = target_actions - noised_actions
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

    def run(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            self._log_params()
            global_step = 0
            pbar = tqdm(range(self.epochs))
            for epoch in pbar:
                epoch_losses = []
                for batch in self.training_loader:
                    loss = self.train_step(*batch)
                    mlflow.log_metric("train/loss", loss, step=global_step)
                    epoch_losses.append(loss)
                    global_step += 1
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                mlflow.log_metric("train/loss_epoch_mean", mean_loss, step=epoch)
                pbar.set_postfix(loss=epoch_losses[-1], epoch=epoch)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "google/paligemma2-3b-pt-224"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pi = SmallPi0.from_pretrained(model_id, device)
    print("Model loaded successfully.")
    dataset_size = 90
    batch_size = 3
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    tokenized_prompts = tokenizer(
        ["drive forward", "go to the living room", "brake"],
        padding=True,
        return_tensors="pt",
    )
    prompt_tokens = tokenized_prompts.input_ids.to(device)
    # prompt_tokens = prompt_tokens.repeat
    prompt_mask = tokenized_prompts.attention_mask.to(device)
    state = torch.randn(batch_size, 2, device=device)
    noised_actions = torch.randn(batch_size, 10, 2, device=device)
    noise_level = torch.rand(batch_size, device=device)
    target_actions = torch.randn(batch_size, 10, 2, device=device)

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
    # output = pi(image, prompt_tokens, prompt_mask, state, noised_actions, noise_level)
    # print(loss)  # should be a scalar value

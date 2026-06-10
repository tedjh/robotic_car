import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

EMBED_DIM = 2
BATCH_SIZE = 64


def design_data_distribution():
    centers = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    cov = 0.1 * torch.eye(EMBED_DIM)
    weights = torch.ones(4) / 4  # torch.softmax(torch.randn(4), dim=0)
    return centers, cov, weights


def sample_source() -> torch.Tensor:
    # Sample from a simple distribution, e.g. Gaussian noise
    return torch.randn(BATCH_SIZE, EMBED_DIM)


def sample_data(
    centers: torch.Tensor, cov: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    # Sample from a mixture of Gaussians
    component = torch.multinomial(weights, BATCH_SIZE, replacement=True)
    return (
        torch.randn(BATCH_SIZE, EMBED_DIM) @ torch.linalg.cholesky(cov)  # pylint: disable=not-callable
        + centers[component]
    )


def plot_loss(losses: list[float], smooth_window: int = 50) -> None:
    plt.plot(losses, alpha=0.3, label="loss")
    if len(losses) >= smooth_window:
        kernel = torch.ones(smooth_window) / smooth_window
        smoothed = torch.conv1d(
            torch.tensor(losses).view(1, 1, -1), kernel.view(1, 1, -1)
        ).flatten()
        plt.plot(
            range(smooth_window - 1, len(losses)),
            smoothed.numpy(),
            label=f"smoothed (w={smooth_window})",
        )
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_samples(
    samples: torch.Tensor,
    true_samples: torch.Tensor,
    centers: torch.Tensor | None = None,
) -> None:
    pts = samples.detach().cpu().numpy()
    plt.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, label="samples")

    pts = true_samples.detach().cpu().numpy()
    plt.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, label="true samples")
    if centers is not None:
        c = centers.cpu().numpy()
        plt.plot(c[:, 0], c[:, 1], "rx", markersize=10, label="centers")
        plt.legend()
    plt.gca().set_aspect("equal")
    plt.show()


def sample_model(model: torch.nn.Module, n_steps: int) -> torch.Tensor:
    x = sample_source()  # start from noise
    ts = torch.linspace(0, 1, n_steps + 1)
    dt = 1.0 / n_steps
    with torch.no_grad():
        for t in ts[:-1]:
            t_batch = t.expand(x.shape[0], 1)
            x = x + dt * model(torch.cat([x, t_batch], dim=1))  # Euler integration
    return x


if __name__ == "__main__":
    centers, cov, weights = design_data_distribution()
    # Example usage
    model = torch.nn.Sequential(
        torch.nn.Linear(EMBED_DIM + 1, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, EMBED_DIM),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses: list[float] = []
    pbar = tqdm(range(10_000))
    for epoch in pbar:
        optimizer.zero_grad()

        # Sample endpoints
        x0 = sample_source()  # e.g. N(0, I)
        x1 = sample_data(centers, cov, weights)  # e.g. a batch from your dataset

        # Sample a timestep
        t = torch.rand(BATCH_SIZE).unsqueeze(1)  # shape (B, 1) for broadcasting

        # Construct x_t by interpolating
        x_t = (1 - t) * x0 + t * x1  # OT-CFM straight-line path
        u_t = x1 - x0  # conditional vector field (constant along path)

        # Regress
        v_pred = model(torch.cat([x_t, t], dim=1))
        loss = (v_pred - u_t).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.4f}")

    plot_loss(losses)

    # Sample from the trained model
    for n_steps in [5, 20, 100]:
        generated_sample = sample_model(model, n_steps=n_steps)
        plot_samples(
            generated_sample,
            true_samples=sample_data(centers, cov, weights),
            centers=centers,
        )

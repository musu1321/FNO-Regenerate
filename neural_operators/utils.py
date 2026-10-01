"""Shared utilities: the relative L2 loss used throughout the paper's experiments,
and a small generic training loop reused by every model notebook."""
import time

import torch


class RelativeL2Loss:
    """Per-sample relative L2 error, ||pred - target||_2 / ||target||_2, averaged over
    the batch. This is the discrete approximation to the Bochner norm error in Eq. (3)
    of the paper, and is the standard metric reported in Section 7."""

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, pred, target):
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        num = torch.norm(pred - target, p=2, dim=1)
        den = torch.norm(target, p=2, dim=1) + self.eps
        return (num / den).mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
    verbose_every=10,
):
    """Generic Adam + cosine-annealing training loop with relative-L2 loss, shared by
    every neural operator model (FNO, GNO, LNO, MGNO) so results are comparable."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = RelativeL2Loss()

    history = {"train_loss": [], "test_loss": []}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss /= n_batches

        model.eval()
        test_loss = 0.0
        n_test_batches = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                n_test_batches += 1
        test_loss /= n_test_batches

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
            elapsed = time.time() - t0
            print(
                f"epoch {epoch:4d} | train rel-L2 {train_loss:.4f} | "
                f"test rel-L2 {test_loss:.4f} | {elapsed:.1f}s"
            )

    return history


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    loss_fn = RelativeL2Loss()
    total = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total += loss_fn(pred, y).item()
        n_batches += 1
    return total / n_batches


@torch.no_grad()
def measure_inference_time(model, x, device="cpu", n_repeats=20):
    model = model.to(device).eval()
    x = x.to(device)
    # warmup
    for _ in range(3):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeats):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n_repeats

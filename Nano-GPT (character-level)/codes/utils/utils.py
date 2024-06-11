import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def guess_loss(model, loader, eval_iters=None):
    model.eval()
    if eval_iters is None:
        eval_iters = 100
    losses = torch.zeros(eval_iters, dtype=torch.float32).to(device)
    for k in range(eval_iters):
        xb, yb = loader.get_batch()
        xb = xb.to(device)
        yb = yb.to(device)
        logits, loss = model(xb, yb)
        losses[k] = loss.item()
    avg_loss = losses.mean().item()
    model.train()
    return avg_loss

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import TrainSeconds, ValSeconds, load_songs
from .model import SecondClassifier


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def train(folder='songs', out='model.pt', epochs=15, batch_size=64, lr=2e-3,
          steps_per_epoch=2000, val_every=5) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    names, signals, rate = load_songs(folder)
    print(f"{len(names)} Songs, {rate} Hz, Gerät: {device}")

    train_ds = TrainSeconds(signals, rate, val_every, length=steps_per_epoch * batch_size)
    val_ds = ValSeconds(signals, rate, val_every)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size)
    print(f"Validierung: {len(val_ds)} Sekunden")

    model = SecondClassifier(len(names), rate).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()
        acc = evaluate(model, val_dl, device) if len(val_ds) else float('nan')
        print(f"Epoche {epoch:2d}  loss {total_loss / steps_per_epoch:.4f}  val-acc {acc:.3f}")
        if acc >= best or len(val_ds) == 0:
            best = acc
            torch.save({'state': model.state_dict(), 'names': names, 'rate': rate}, out)
    print(f"Bestes Modell ({best:.3f}) gespeichert: {out}")

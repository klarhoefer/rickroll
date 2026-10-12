import torch
from torch import nn
import torchaudio


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SecondClassifier(nn.Module):
    """1 Sekunde Audio (B, T) -> Log-Mel-Spektrogramm -> kleines CNN -> Logits pro Song."""

    def __init__(self, n_songs: int, rate: int, n_mels: int = 64):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=rate, n_fft=1024, hop_length=256, n_mels=n_mels)
        self.cnn = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, n_songs)

    def forward(self, x):
        spec = torch.log(self.mel(x) + 1e-6).unsqueeze(1)  # (B,1,mel,frames)
        # Pro Beispiel standardisieren -> unabhängig von der Lautstärke
        spec = (spec - spec.mean((1, 2, 3), keepdim=True)) / (spec.std((1, 2, 3), keepdim=True) + 1e-5)
        h = self.cnn(spec).mean((2, 3))
        return self.fc(self.drop(h))

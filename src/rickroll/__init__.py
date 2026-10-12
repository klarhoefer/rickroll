import os

from .chunks import WavFile

def main() -> None:
    wavs = os.listdir('songs')
    for wav in wavs:
        wf = WavFile(os.path.join('songs', wav))
        print(wf.get_second(0))  # Beispiel: Hole die Daten der ersten Sekunde

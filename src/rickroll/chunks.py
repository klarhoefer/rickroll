import wave
import numpy as np

class WavFile:
    def __init__(self, file_path):
        self.file_path = file_path
        with wave.open(file_path, 'rb') as w:
            breite = w.getsampwidth()
            self.rate = w.getframerate()
            roh = w.readframes(w.getnframes())

            # Datentyp je nach Bittiefe
            if breite == 1:
                daten = np.frombuffer(roh, dtype=np.uint8).astype(np.float32)
                daten /= 255.0                       # 8 Bit ist unsigned: 0..255 -> 0..1
            elif breite == 2:
                daten = np.frombuffer(roh, dtype=np.int16).astype(np.float32)
                daten = (daten + 32768) / 65535.0    # -32768..32767 -> 0..1
            elif breite == 4:
                daten = np.frombuffer(roh, dtype=np.int32).astype(np.float64)
                daten = (daten + 2**31) / (2**32 - 1)
            else:
                raise ValueError(f"Nicht unterstützte Sample-Breite: {breite} Bytes")

            self.daten = daten

    def get_second(self, second: int):
        start = second * self.rate
        end = (second + 1) * self.rate
        return self.daten[start:end]

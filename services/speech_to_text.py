import os
import time

import numpy as np
import torch
import whisper


class WhisperSerice:

    def __init__(self, model_size="base"):
        self.model_size = model_size

        try:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"

            self.model = whisper.load_model(model_size, device=self.device)

        except Exception as e:
            print(f"Error loading Whisper model: {e}")

    def transcribe(self, audio_data):

        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()

        if audio_data.dtype != "float32":
            audio_data = audio_data.astype(np.float32)

        is_fp16 = self.device == "cuda"
        result = self.model.transcribe(audio_data, fp16=is_fp16)

        return result["text"].strip()


if __name__ == "__main__":
    whisper_service = WhisperSerice(model_size="tiny")
    audio_path = "ref_.wav"
    s = time.time()
    output = whisper_service.model.transcribe(audio_path)
    e = time.time()
    print(f"Transcription took {e - s:.2f} seconds")
    print("Transcription:", output["text"].strip())

import numpy as np
import torch
import tqdm

from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)

def get_audio_emb(model_name, device, audio):
    from whisper import load_model
    model = load_model(model_name, device=device)
    mel = log_mel_spectrogram(audio, padding=N_SAMPLES).cuda()
    mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device)
    mel_segment = mel_segment.unsqueeze(0)
    print(mel_segment.shape)
    return model.embed_audio(mel_segment)

if __name__ == "__main__":
    emb = get_audio_emb("large", "cuda:0", "/root/eason/test_translation.m4a")
    emb = torch.squeeze(emb)
    print(emb)
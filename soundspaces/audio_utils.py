import numpy as np
import torch
from torchaudio.functional import amplitude_to_DB, melscale_fbanks
from typing import Optional

def compute_spectrogram(
    audio_data, win_length: int, hop_length: int, n_fft: int, n_mels: int,
    window: Optional[torch.Tensor], mel_scale: Optional[torch.Tensor],
    downsample: Optional[int], include_gcc_phat: bool, backend: str = "torch",
):
    assert backend in ("torch", "numpy")
    # stft.shape = (C=2, T, F)
    stft = torch.stack(
        [
            torch.stft(
                input=torch.tensor(X_ch, device='cpu', dtype=torch.float32),
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
                center=True,
                window=(
                    window if window is not None
                    else torch.hann_window(win_length, device="cpu")
                ),
                pad_mode="constant", # constant for zero padding
                return_complex=True,
            ).T
            for X_ch in audio_data
        ],
        dim=0
    )
    # Compute power spectrogram
    spectrogram = (torch.abs(stft) ** 2.0).to(dtype=torch.float32)
    # Apply the mel-scale filter to the power spectrogram
    if mel_scale is not None:
        spectrogram = torch.matmul(spectrogram, mel_scale)
    # Optionally downsample
    if downsample:
        spectrogram = torch.nn.functional.avg_pool2d(
            spectrogram.unsqueeze(dim=0),
            kernel_size=(downsample, downsample),
        ).squeeze(dim=0)
    # Convert to decibels
    spectrogram = amplitude_to_DB(
        spectrogram,
        multiplier=20.0,
        amin=1e-10,
        db_multiplier=0.0,
        top_db=80,
    )

    if include_gcc_phat:
        num_channels = stft.shape[0]
        # compute gcc_phat : (comb, T, F)
        out_list = []
        for ch1 in range(num_channels - 1):
            for ch2 in range(ch1 + 1, num_channels):
                x1 = stft[ch1]
                x2 = stft[ch2]
                xcc = torch.angle(x1 * torch.conj(x2))
                xcc = torch.exp(1j * xcc.type(torch.complex64))
                gcc_phat = torch.fft.irfft(xcc)
                # Just get a subset of GCC values to match dimensionality
                gcc_phat = torch.cat(
                    [
                        gcc_phat[..., -n_mels // 2:],
                        gcc_phat[..., :n_mels // 2],
                    ],
                    dim=-1,
                )
                out_list.append(gcc_phat)
        gcc_phat = torch.stack(out_list, dim=0)

        # Downsample
        if downsample:
            gcc_phat = torch.nn.functional.avg_pool2d(
                gcc_phat,
                kernel_size=(downsample, downsample),
            )

        # spectrogram.shape = (C=3, T, F)
        spectrogram = torch.cat([spectrogram, gcc_phat], dim=0)

    # Reshape to how SoundSpaces expects
    # spectrogram.shape = (F, T, C)
    spectrogram = spectrogram.permute(2, 1, 0)
    if backend == "torch":
        return spectrogram
    elif backend == "numpy":
        return spectrogram.numpy().astype(np.float32)
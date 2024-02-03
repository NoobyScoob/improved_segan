import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from math import floor
import os
import time


class SpeechDataset(Dataset):
    def __init__(self, clean_folder, noise_folder):
        self.clean_folder = clean_folder
        self.noise_folder = noise_folder

        self.clean_files = os.listdir(clean_folder)
        self.noise_files = os.listdir(noise_folder)

        self.clean_files.sort()
        self.noise_files.sort()

        self.segments = []
        self.size = 0

        self.next_idx = 0
        self.window_size = 16384
        self.hop_length = self.window_size//2
    
    def __len__(self):
        if self.size:
            return self.size
        
        self.size = 0
        for f in self.clean_files:
            clean_path = os.path.join(self.clean_folder, f)
            clean_waveform, _ = torchaudio.load(clean_path, normalize=True)
            
            start_idx = 0
            while start_idx < clean_waveform.size(1):
                self.size += 1
                start_idx += self.hop_length
            
        return self.size

    
    def __getitem__(self, i):
        if i < len(self.segments):
            return self.segments[i] # check memory footprint
        
        clean_path = os.path.join(self.clean_folder, self.clean_files[self.next_idx])
        noise_path = os.path.join(self.noise_folder, self.noise_files[self.next_idx])

        self.next_idx += 1

        clean_waveform, _ = torchaudio.load(clean_path, normalize=True)
        noise_waveform, _ = torchaudio.load(noise_path, normalize=True)

        start_idx = 0
        while start_idx < clean_waveform.size(1):
            clean_segment = clean_waveform[:, start_idx:start_idx + self.window_size]
            noise_segment = noise_waveform[:, start_idx:start_idx + self.window_size]

            if clean_waveform.size(1) - start_idx < self.window_size:
                add_len = self.window_size - (clean_waveform.size(1) - start_idx)
                clean_segment = torch.cat([clean_segment, clean_waveform[:, :add_len]], dim=1)
                noise_segment = torch.cat([noise_segment, noise_waveform[:, :add_len]], dim=1)

            self.segments.append((noise_segment.T, clean_segment.T))
            start_idx += self.hop_length
        
        return self.segments[i]


def d1_loss(d_outputs, reduction="mean"):
    """Calculates the loss of the discriminator when the inputs are clean    """
    output = 0.5 * ((d_outputs - 1) ** 2)
    if reduction == "mean":
        return output.mean()
    elif reduction == "batch":
        return output.view(output.size(0), -1).mean(1)
    

def d2_loss(d_outputs, reduction="mean"):
    """Calculates the loss of the discriminator when the inputs are not clean    """
    output = 0.5 * ((d_outputs) ** 2)
    if reduction == "mean":
        return output.mean()
    elif reduction == "batch":
        return output.view(output.size(0), -1).mean(1)
    

def g3_loss(
    d_outputs,
    predictions,
    targets,
    length,
    l1LossCoeff,
    klLossCoeff,
    z_mean=None,
    z_logvar=None,
    reduction="mean",
):
    """Calculates the loss of the generator given the discriminator outputs    """
    discrimloss = 0.5 * ((d_outputs - 1) ** 2)
    l1norm = torch.nn.functional.l1_loss(predictions, targets, reduction="none")

    if not (
        z_mean is None
    ):  # This will determine if model is being trained as a vae
        ZERO = torch.zeros_like(z_mean)
        distq = torch.distributions.normal.Normal(
            z_mean, torch.exp(z_logvar) ** (1 / 2)
        )
        distp = torch.distributions.normal.Normal(
            ZERO, torch.exp(ZERO) ** (1 / 2)
        )
        kl = torch.distributions.kl.kl_divergence(distq, distp)
        kl = kl.sum(axis=1).sum(axis=1).mean()
    else:
        kl = 0
    if reduction == "mean":
        return (
            discrimloss.mean() + l1LossCoeff * l1norm.mean() + klLossCoeff * kl
        )
    elif reduction == "batch":
        dloss = discrimloss.view(discrimloss.size(0), -1).mean(1)
        lloss = l1norm.view(l1norm.size(0), -1).mean(1)
        return dloss + l1LossCoeff * lloss + klLossCoeff * kl
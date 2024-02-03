import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from math import floor
import os
import time
from utils import SpeechDataset, d1_loss, d2_loss, g3_loss
from attention import SelfAttention
from gan import Generator, Discriminator

clean_folder = '../data/clean_trainset_28spk_wav/'
noise_folder = '../data/noisy_trainset_28spk_wav/'

batch_size = 100
dataset = SpeechDataset(clean_folder, noise_folder)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if __name__ == "__main__":
    
    generator = Generator(5, SelfAttention, attention_positions=[10]).to(device)
    discriminator = Discriminator(5, SelfAttention, attention_positions=[10]).to(device)

    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    num_epochs = 50
    log_interval = 100

    for epoch in range(num_epochs):
    
        counter = 0
        start = time.time()
        for i, (noisy_speech, clean_speech) in enumerate(dataloader):
            
            # Move data to the appropriate device (GPU or CPU)
            noisy_speech, clean_speech = noisy_speech.to(device), clean_speech.to(device)

            optimizer_d.zero_grad()

            # Train Discriminator
            optimizer_d.zero_grad()
            generated_speech = generator(noisy_speech)
            generated_speech = generated_speech.detach()

            fake_inputs = torch.cat([generated_speech, noisy_speech], dim=2)
            d_outputs_generated = discriminator(fake_inputs) # Check Fake - Classify as Zero
            
            real_inputs = torch.cat([clean_speech, noisy_speech], dim=2) # check this
            d_outputs_real = discriminator(real_inputs) # Check Real - Classify as One
            
            loss_d = d1_loss(d_outputs_real) + d2_loss(d_outputs_generated)
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            denoised_speech = generator(noisy_speech)
            d_inputs_fake = torch.cat([denoised_speech, clean_speech], dim=2)
            d_outputs_fake = discriminator(d_inputs_fake)
            loss_g = g3_loss(d_outputs_fake, denoised_speech, clean_speech, noisy_speech.size(1), 0, 0)
            loss_g.backward()
            optimizer_g.step()

            # Print training information
            if i % log_interval == 0:
                print(f"Epoch {epoch}/{num_epochs}, Batch {i}/{len(dataloader)}, Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}, Time: {time.time()-start}")

    
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
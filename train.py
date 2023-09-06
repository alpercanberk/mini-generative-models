import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vae import VAE
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def visualize_generation(model, latent_dim, ax):
    """Visualizes generated images from the VAE model"""
    with torch.no_grad():
        # Sample random latent variables (assuming Gaussian prior)
        z = torch.randn(16, latent_dim)  # 16 for a 4x4 grid

        # Generate images using the decoder
        generated_images = model.decode(z)

        # Rescale images to [0, 1] from [-1, 1]
        generated_images = (generated_images + 1) / 2.0

        # Convert to numpy array and reshape
        np_images = generated_images.cpu().numpy().transpose((0, 2, 3, 1))

        # Stitch images into a grid
        grid_img = np.zeros((4 * 32, 4 * 32, 3))
        for i in range(4):
            for j in range(4):
                grid_img[i*32:(i+1)*32, j*32:(j+1)*32, :] = np_images[i * 4 + j]

        ax.imshow(grid_img)
        ax.axis('off')


def main():
    # Parameters
    n_epochs = 1000
    batch_size = 32
    input_dims = (3, 32, 32)  # CIFAR-100 images are 32x32 RGB
    latent_dim = 128
    lr = 1e-3
    n_iters = None

    # Data loading
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer and loss function
    model = VAE(input_dims, enc_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    plt.ion()
    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("Reconstruction Loss")
    ax[1].set_title("KL Loss")

    recon_losses = []
    kl_losses = []

    # tqdm bar for epochs
    epoch_pbar = tqdm(range(n_epochs), desc="Epochs")

    for epoch in epoch_pbar:
        model.train()

        # Initialize running loss values for the epoch
        running_recon_loss = 0.0
        running_kl_loss = 0.0

        # tqdm bar for batches
        batch_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches", leave=False)

        for batch_idx, (data, _) in batch_pbar:
            if n_iters and batch_idx >= n_iters:
                break

            optimizer.zero_grad()

            # Forward pass
            model_output = model(data)  # returns x_hat, mean, logvar

            # Calculate loss
            loss, info = VAE.loss_function(*model_output, data)
            recon_loss = info['recon_loss']
            kl_loss = info['kl_loss']

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss values
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_loss.item()

            # Update tqdm postfix
            batch_pbar.set_postfix({
                'Recon Loss': f'{recon_loss.item():.4f}',
                'KL Loss': f'{kl_loss.item():.4f}'
            })

            if batch_idx % 32 == 0:
                recon_losses.append(recon_loss.item())
                kl_losses.append(kl_loss.item())
                ax[0].cla()
                ax[1].cla()
                ax[0].plot(recon_losses, label='Recon Loss')
                ax[1].plot(kl_losses, label='KL Loss')
                ax[0].legend()
                ax[1].legend()
                plt.pause(0.01)

            if batch_idx % 64 == 0:
                visualize_generation(model, latent_dim, ax[2])

        # Calculate and display average loss values for the epoch
        avg_recon_loss = running_recon_loss / len(train_loader)
        avg_kl_loss = running_kl_loss / len(train_loader)

        epoch_pbar.set_postfix({
            'Avg Recon Loss': f'{avg_recon_loss:.4f}',
            'Avg KL Loss': f'{avg_kl_loss:.4f}'
        })

    plt.ioff()  # Turn interactive mode off at the end
    plt.show()

if __name__ == '__main__':
    main()

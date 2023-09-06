import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class VAEEncoderLinear(nn.Module):
    def __init__(self, input_shape, enc_dim, hidden_size=512):

        super().__init__()

        self.fc1 = nn.Linear(np.prod(input_shape), enc_dim)
        self.fc2_hidden = nn.Linear(enc_dim, hidden_size)  # Additional hidden layer
        self.fc2_mean = nn.Linear(hidden_size, enc_dim)
        self.fc2_logvar = nn.Linear(hidden_size, enc_dim)

    def transform_in(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):

        x = self.transform_in(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2_hidden(x)  # Apply the additional hidden layer
        x = F.relu(x)

        mu = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)

        return mu, logvar
    
class VAEDecoderLinear(nn.Module):

    def __init__(self, input_shape, enc_dim, hidden_size=512):

        super().__init__()

        self.fc1 = nn.Linear(enc_dim, hidden_size)  # Additional hidden layer
        self.fc2 = nn.Linear(hidden_size, np.prod(input_shape))

        self.input_shape = input_shape

    def transform_out(self, x):
        return x.view(-1, *self.input_shape)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        X = F.relu(x)
        x = self.transform_out(x)
        return x
    
class VAEEncoderConv(nn.Module):
    def __init__(self, input_dims, enc_dim, hidden_size=512):
        super().__init__()

        self.enc_dim = enc_dim
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc_hidden = nn.Linear(128 * 4 * 4, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, enc_dim)
        self.fc_logvar = nn.Linear(hidden_size, enc_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_hidden(x))
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class VAEDecoderConv(nn.Module):
    def __init__(self, input_dims, enc_dim):
        super().__init__()

        self.enc_dim = enc_dim

        # Fully connected layer to reshape the input
        self.fc1 = nn.Linear(enc_dim, 256 * 4 * 4)  

        # Transposed Convolutional Layers with more channels
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(64, input_dims[0], kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), 256, 4, 4)  # Reshape to [batch_size, 256, 4, 4]

        x = F.relu(self.deconv1(x))  # Output shape: [batch_size, 256, 4, 4]
        x = F.relu(self.deconv2(x))  # Output shape: [batch_size, 128, 8, 8]
        x = F.relu(self.deconv3(x))  # Output shape: [batch_size, 128, 8, 8]
        x = F.relu(self.deconv4(x))  # Output shape: [batch_size, 64, 16, 16]
        x = F.relu(self.deconv5(x))  # Output shape: [batch_size, 64, 16, 16]
        x = torch.tanh(self.deconv6(x))  # Output shape: [batch_size, 3, 32, 32]

        return x

class VAE(nn.Module):
    def __init__(self, input_shape, enc_dim, net_type='conv'):
        super(VAE, self).__init__()

        if net_type == 'dense':
            self.encoder = VAEEncoderLinear(input_shape, enc_dim)
            self.decoder = VAEDecoderLinear(input_shape, enc_dim) 
        elif net_type == 'conv':
            self.encoder = VAEEncoderConv(input_shape, enc_dim)
            self.decoder = VAEDecoderConv(input_shape, enc_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)

        eps = torch.randn_like(logvar) #sample unit gaussians1
        z = torch.exp(0.5 * logvar) * eps + mean #reparameterization trick
        x_hat = self.decoder(z)

        return x_hat, mean, logvar
    
    @staticmethod
    def loss_function(x_hat, mean, logvar, x):


        assert x_hat.size() == x.size(), f"Shape of x_hat: {x_hat.size()} and x: {x.size()} do not match"

        #reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)
        
        #KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss, {'recon_loss': recon_loss, 
                                      'kl_loss': kl_loss, 
                                      'total': recon_loss + kl_loss}
    def decode(self, z):
        return self.decoder(z)
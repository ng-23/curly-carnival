import torch
from torch import nn
import numpy as np

registered_generators = {}

registered_discriminators = {}

def register_generator(name):
    def wrapper(cls):
        registered_generators[name] = cls
        return cls
    return wrapper

def register_discriminator(name):
    def wrapper(cls):
        registered_discriminators[name] = cls
        return cls
    return wrapper

def get_generator(name:str, **model_kwargs) -> nn.Module:
    name = name.upper()

    if name not in registered_generators:
        raise ValueError(f'No generator is registered with name {name}')
    
    generator_class = registered_generators[name]
    return generator_class(**model_kwargs)

def get_discriminator(name:str, **model_kwargs) -> nn.Module:
    name = name.upper()

    if name not in registered_discriminators:
        raise ValueError(f'No discriminator is registered with name {name}')
    
    discriminator_class = registered_discriminators[name]
    return discriminator_class(**model_kwargs)

@register_generator('DCGAN')
class DCGANGenerator(nn.Module):
    '''
    Taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    with minor modifications

    Also see https://github.com/Lornatang/DCGAN-PyTorch/blob/master/model.py
    '''

    IMG_SIZE = 64
    NUM_CHANNELS = 3

    def __init__(self, latent_size=100, **kwargs):
        super(DCGANGenerator, self).__init__()

        self.latent_size = latent_size

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.latent_size, self.IMG_SIZE * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.IMG_SIZE * 8),
            nn.ReLU(True),
            # state size. ``(IMG_SIZE*8) x 4 x 4``
            nn.ConvTranspose2d(self.IMG_SIZE * 8, self.IMG_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.IMG_SIZE * 4),
            nn.ReLU(True),
            # state size. ``(IMG_SIZE*4) x 8 x 8``
            nn.ConvTranspose2d( self.IMG_SIZE * 4, self.IMG_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.IMG_SIZE * 2),
            nn.ReLU(True),
            # state size. ``(IMG_SIZE*2) x 16 x 16``
            nn.ConvTranspose2d( self.IMG_SIZE * 2, self.IMG_SIZE, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.IMG_SIZE),
            nn.ReLU(True),
            # state size. ``(IMG_SIZE) x 32 x 32``
            nn.ConvTranspose2d( self.IMG_SIZE, self.NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(NUM_CHANNELS) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

@register_discriminator('DCGAN')
class DCGANDiscriminator(nn.Module):
    '''
    Taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    with minor modifications

    Also see https://github.com/Lornatang/DCGAN-PyTorch/blob/master/model.py
    '''

    IMG_SIZE = 64
    NUM_CHANNELS = 3
    
    def __init__(self, **kwargs):
        super(DCGANDiscriminator, self).__init__()

        self.main = nn.Sequential(
            # input is ``(NUM_CHANNELS) x 64 x 64``
            nn.Conv2d(self.NUM_CHANNELS, self.IMG_SIZE, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(IMG_SIZE) x 32 x 32``
            nn.Conv2d(self.IMG_SIZE, self.IMG_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.IMG_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(IMG_SIZE*2) x 16 x 16``
            nn.Conv2d(self.IMG_SIZE * 2, self.IMG_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.IMG_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(IMG_SIZE*4) x 8 x 8``
            nn.Conv2d(self.IMG_SIZE * 4, self.IMG_SIZE * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.IMG_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(IMG_SIZE*8) x 4 x 4``
            nn.Conv2d(self.IMG_SIZE * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

@register_generator('VANILLA')  
class VanillaGenerator(nn.Module):
    '''
    Code largely taken and adapted from https://www.baeldung.com/cs/pytorch-generative-adversarial-networks#:~:text=2.%20GAN
    '''

    IMG_SIZE=64
    NUM_CHANNELS=3

    def __init__(self, latent_size=100, hidden_size=512, **kwargs):
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.IMG_SIZE*self.IMG_SIZE*self.NUM_CHANNELS),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, img_size*img_size*num_channels)
        return self.main(x)

@register_discriminator('VANILLA')
class VanillaDiscriminator(nn.Module):
    '''
    Code largely taken and adapted from https://www.baeldung.com/cs/pytorch-generative-adversarial-networks#:~:text=2.%20GAN
    '''

    IMG_SIZE = 64
    NUM_CHANNELS = 3

    def __init__(self, hidden_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.main = nn.Sequential(
            nn.Linear(self.IMG_SIZE*self.IMG_SIZE*self.NUM_CHANNELS, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, img_size*img_size*num_channels)
        return self.main(x)

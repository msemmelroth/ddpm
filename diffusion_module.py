import torch
from torch import nn

class diffusion(nn.Module):
    def __init__(self, startVariance, maxVariance, diffusionSteps, spacing_func = None, device = 'cpu'):
        super().__init__()
        self.startVariance = startVariance
        self.maxVariance = maxVariance
        self.diffusionSteps = diffusionSteps
        self.device = device

        #Initializing hyperparameter(s) for faster calculations
        if spacing_func is None:
            betas = torch.linspace(self.startVariance, self.maxVariance, diffusionSteps).to(self.device)
        else:
            betas = spacing_func().to(self.device)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1-self.betas)
        self.register_buffer("alphaBar", torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrtAlphaBar', torch.sqrt(self.alphaBar))
        self.register_buffer('oneMinusSqrtAlphaBar', torch.sqrt(1-self.alphaBar))

    def forward(self, input, noise, timestep):
        """
        Input is (B,C,H,W): (Batch size, Channel, Height, Width), batch size number of images, each with C channels, and HxW resolution
        Reshape till (B,) becomes (B,1,1,1)
        These extra singleton dimensions allow for "broadcasting" for the element wise calculations between 
        the alpha-variable tensors and the input image, they would likely initially have different shapes
        """

        sqrtAlphaBar = self.sqrtAlphaBar[timestep][:, None, None, None]
        oneMinusSqrtAlphaBar = self.oneMinusSqrtAlphaBar[timestep][:, None, None, None]
        noisedInput = (sqrtAlphaBar*input)+oneMinusSqrtAlphaBar*noise

        return noisedInput

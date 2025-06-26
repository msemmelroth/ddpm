import torch
from torch import nn

class VarianceScheduler(nn.Module):
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
    



#put in sampling function
def ddpmReverse(model, noise_module, num_samples, channels, img_size, prediction_type):
    model.eval()
    with torch.inference_mode():
        x = torch.randn((num_samples, channels, img_size, img_size)).to(noise_module.device)
        for i in reversed(range(noise_module.diffusionSteps)):
            t = (torch.ones(num_samples) * i).long().to(noise_module.device)
            model_prediction = model(x, t)
            alphas = noise_module.alphas[t][:, None, None, None]
            alpha_bar = noise_module.alphaBar[t][:, None, None, None]
            betas = noise_module.betas[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)



            #TODO: Double Check these equations, find where they come from etc
            if prediction_type == 'image':
                # Compute mean for x_{t-1} using x0 prediction
                mean = (1 / torch.sqrt(alphas)) * (
                    x - (betas / torch.sqrt(1 - alpha_bar)) * (x - model_prediction)
                )
                x = mean + torch.sqrt(betas) * noise

            elif prediction_type == 'noise':
                x = 1 / torch.sqrt(alphas) * (x - ((1 - alphas) / (torch.sqrt(1 - alpha_bar))) * model_prediction) +\
                    torch.sqrt(betas) * noise
                
            else:
                raise ValueError('Value for "prediction_type" must be "noise" or "image". ')
    

    return x

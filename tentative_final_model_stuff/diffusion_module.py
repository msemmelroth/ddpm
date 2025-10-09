import torch
from torch import nn

class diffusion(nn.Module):
    def __init__(self, startVariance, maxVariance, diffusionSteps, spacing=1, device = 'cpu'):
        super().__init__()
        
        self.startVariance = startVariance
        self.maxVariance = maxVariance
        self.diffusionSteps = diffusionSteps
        self.spacing = spacing
        self.device = device

        #Initializing hyperparameter(s) for faster calculations
        if type(spacing) in (int, float):
            t = torch.linspace(0, 1, diffusionSteps).to(self.device)
            betas = self.startVariance + (self.maxVariance - self.startVariance) * t ** self.spacing
            #beta = b_0 + (b_T-b_0)*t^spacing

        #add hyperprams to buffer for saving, important to know the exact schedule used
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1-self.betas)
        self.register_buffer('alphaBar', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrtAlphaBar', torch.sqrt(self.alphaBar))
        self.register_buffer('oneMinusSqrtAlphaBar', torch.sqrt(1-self.alphaBar))

    def forward(self, input, noise, timestep):
        """
        Input is (B,C,H,W): (Batch size, Channel, Height, Width), batch size number of images, each with C channels, and HxW resolution
        Reshape till (B,) becomes (B,1,1,1)
        These extra singleton dimensions allow for "broadcasting" for the element wise calculations between 
        the alpha-variable tensors and the input image, as they initially have different shapes
        """
        
        #broadcast the hyperparameters to the dimensions of the feature map 1D Tensor to a 4D (B,C,H,W) shaped tensor
        sqrtAlphaBar = self.sqrtAlphaBar[timestep][:, None, None, None]
        oneMinusSqrtAlphaBar = self.oneMinusSqrtAlphaBar[timestep][:, None, None, None]
        
        noisedInput = (sqrtAlphaBar*input)+oneMinusSqrtAlphaBar*noise

        return noisedInput
    
    def sampling(self, model, num_samples, channels, img_size, prediction_type):
        model.eval()
        with torch.inference_mode(): #No gradient calculation
            x = torch.randn((num_samples, channels, img_size[0], img_size[1])).to(self.device)
            for i in reversed(range(self.diffusionSteps)):
                t = (torch.ones(num_samples) * i).long().to(self.device)
                
                model_prediction = model(x, t)
                alphas = self.alphas[t][:, None, None, None]
                alpha_bar = self.alphaBar[t][:, None, None, None]
                betas = self.betas[t][:, None, None, None]

                if i > 1: 
                    noise = torch.randn_like(x) #z ~ N(0,I)
                else: 
                    noise = torch.zeros_like(x) #no noise on the last time step


                if prediction_type == 'image':
                    # Compute mean for x_{t-1} using x0 prediction

                    #mean equation is equation 11 in DDPM paper
                    mean = (1 / torch.sqrt(alphas)) * (
                        x - (betas / torch.sqrt(1 - alpha_bar)) * (x - model_prediction) #(x - model prediction of image) is equivalent to a model prediction of noise epsilon_theta
                    )
                    x = mean + torch.sqrt(betas) * noise


                elif prediction_type == 'noise': #Equation found in ddpm paper between Eqs 11 and 12
                    x = 1 / torch.sqrt(alphas) * (x - ((1 - alphas) / (torch.sqrt(1 - alpha_bar))) * model_prediction) +\
                        torch.sqrt(betas) * noise
                    
                else:
                    raise ValueError('Value for prediction_type must be "noise" or "image".')

        return x

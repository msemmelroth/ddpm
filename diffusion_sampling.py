import torch
def ddpmReverse(model, noise_module, num_samples, channels, img_size, prediction_type):
    model.eval()
    with torch.inference_mode():
        x = torch.randn((num_samples, channels, img_size[0], img_size[1])).to(noise_module.device)
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
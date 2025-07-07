from torch import nn
from torch import cuda

initial_channel_count = 2 #number of channels the initial image has, possibly add an output channel count if we want a different output
img_size = (128, 128)
images_path = 'mc_outputs'

if cuda.is_available(): dvice = 'cuda'
else: device = 'cpu'


UNet_params = {
    "scale_rate" : 3, #integer controlling the amount the spatial resolution of the feature map decreases/increases during the down/upsampling process
    "t_dim" : 128, #integer controlling dimensionality of positional embedding vector of the timesteps
    "num_conv_blocks" : 1 #integer that controls how many convolutional blocks there are per up/down sampling block
    # ,"use_attention" : True
}

diffusion_parameters = {
    'schedule_spacing' : 1/4, #integer for exponent spacing
    "num_timesteps" : 1000,
    "max_variance" : 0.02,
}

training_params = {
    'batch_size' : 16,
    'num_epochs' : 5,
    'fmap_lossfunc': nn.MSELoss(),
    'sbit_lossfunc': nn.BCEWithLogitsLoss(),
    'prediction_type' : 'noise' #noise or image, training must be redone if sampling for the opposite
}




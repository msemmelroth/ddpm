from torch import nn
from torch import cuda

initial_channel_count = 3 #number of channels the initial image has plus the sparstiy bit channel =3 right now, R, G, sparsity mask
img_size = (20, 20)
images_path = 'mc_outputs'

if cuda.is_available(): dvice = 'cuda'
else: device = 'cpu'


UNet_params = {
    "scale_rate" : 3, #integer controlling the amount the spatial resolution of the feature map decreases/increases during the down/upsampling process
    "t_dim" : 128, #integer controlling dimensionality of positional embedding vector of the timesteps
    "num_conv_blocks" : 2 #integer that controls how many convolutional blocks there are per up/down sampling block, 2 for double conv as in unet paper
}

diffusion_params = {
    'schedule_spacing' : 1, #integer for power/root spacing
    "num_timesteps" : 1000,
    "max_variance" : 0.02,
}

training_params = {
    'batch_size' : 10,
    'num_epochs' : 1,
    'fmap_lossfunc': nn.MSELoss(),
    'sbit_lossfunc': nn.BCEWithLogitsLoss(),
    'prediction_type' : 'noise' #noise or image
}



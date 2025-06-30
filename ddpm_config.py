from torch import nn
initial_channel_count = 2 #number of channels the initial image has, possibly add an output channel count if we want a different output
img_size = (128, 128)
images_path = 'mc_outputs'

UNet_params = {
    "scale_rate" : 4, #integer controlling the amount the spatial resolution of the feature map decreases/increases during the down/upsampling process
    "time_embedding_dimensions" : 128, #integer controlling dimensionality of positional embedding vector of the timesteps
    "num_conv_blocks_per_scaling_block" : 1, #integer that controls how many convolutional blocks there are per up/down sampling block
    "use_attention" : True,
}

diffusion_parameters = {
    'scheduling_type' : 'linear',
    "num_timesteps" : 1000,
    "max_variance" : 0.02,
}

training_params = {
    'batch_size' : 1,
    'num_epochs' : 20,
    'fmap_lossfunc': nn.MSELoss(),
    'sbit_lossfunc': nn.BCEWithLogitsLoss()
}




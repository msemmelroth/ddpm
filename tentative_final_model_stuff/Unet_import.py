import torch
from torch import nn
from ddpm_config import UNet_params, initial_channel_count

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual = True, use_mid_ch = False):
        super().__init__() 
        self.use_residual = use_residual

        self.convBlocks = nn.ModuleList()

        # add intermediate convolutional blocks with output channels not changing from input ones
        for _ in range(UNet_params["num_conv_blocks"]-1): #-1 because we need to append the final conv block, use -2 if we need the 2nd to last to be special, e.g: when we use mid channels
            
            #all conv blocks will be in channels to in channels except for the last one
            self.convBlocks.append( nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(num_groups=3, num_channels=in_channels),
            nn.SiLU()
        ))
            
        #append the final convolutional block with out_channels as the output
        self.convBlocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(num_groups=3, num_channels=out_channels),
            nn.SiLU()))

        #accounts for any mismatching channels during addition of residual
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    
    def forward(self, x):
        residual = self.residual(x)
        for block in self.convBlocks:
            x = block(x)

        if self.use_residual: 
            return residual + x
        else: return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dimensions):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling for downsampling
        self.residualBlock = ResidualBlock(in_channels=in_channels, out_channels=out_channels, use_residual=False)
    
        self.t_silu = nn.SiLU()
        self.t_linear = nn.Linear(in_features=t_dimensions, out_features=out_channels)

    def forward(self, x, t):
        x = self.pool(x)
        x = self.residualBlock(x)
        
        t_embed = self.t_silu(t)
        t_embed = self.t_linear(t_embed)
        t_embed = t_embed[:,:,None,None].repeat(1,1,x.shape[2], x.shape[3])
        
        return x + t_embed


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dimensions):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_dimensions = t_dimensions

        self.residualBlock = ResidualBlock(in_channels=in_channels, out_channels=out_channels, use_residual=True)

        self.t_silu = nn.SiLU()
        self.t_linear = nn.Linear(in_features=t_dimensions, out_features=out_channels)

    def forward(self, x, skip, t):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1) #concatenate the cross bottleneck output with current output along their 2nd of 4 dimensions

        x = self.residualBlock(x)
    
        t_embed = self.t_silu(t)
        t_embed = self.t_linear(t_embed)

        t_embed = t_embed[:,:,None,None].repeat(1,1,x.shape[2], x.shape[3])
        
        return x + t_embed
    

class UNet(nn.Module):
    def __init__(self, t_dim=UNet_params["t_dim"], device='cpu'):
        super(UNet, self).__init__()

        self.device = device
        self.t_dim = t_dim

        icc = initial_channel_count #from config file
        scale_rate = UNet_params["scale_rate"]
        f_inc = 30 #number of channels for first convolution to increase to


        #                    ---UNET Architecture Components---
        self.input = ResidualBlock(in_channels=icc, out_channels=f_inc, use_mid_ch=False, use_residual=True)

        self.d1 = DownsampleBlock(in_channels=f_inc, out_channels=f_inc*scale_rate, t_dimensions=self.t_dim)
        self.d2 = DownsampleBlock(in_channels=f_inc*scale_rate, out_channels=f_inc*scale_rate, t_dimensions=self.t_dim)

        self.bottom1 = ResidualBlock(in_channels=f_inc*scale_rate, out_channels=f_inc*scale_rate**2, use_residual = False)
        self.bottom2 = ResidualBlock(in_channels=f_inc*scale_rate**2, out_channels=f_inc*scale_rate**2, use_residual = False)
        self.bottom3 = ResidualBlock(in_channels=f_inc*scale_rate**2, out_channels=f_inc*scale_rate, use_residual = False)

        self.u1 = UpsampleBlock(in_channels=(f_inc*scale_rate)*2, out_channels=(f_inc), t_dimensions=self.t_dim)
        self.u2 = UpsampleBlock(in_channels=f_inc*2, out_channels=(f_inc), t_dimensions=self.t_dim)

        self.out = nn.Conv2d(in_channels=f_inc, out_channels=icc, kernel_size=1)

    def position_embeddings(self, t, t_dim_channels):
        #Creates a tensor for the frequency denominator of length channels/2 (because step=2 and we want half to be even embeddings and half to be odd embeddings) and normalizes it to the number of total embedding channels
        freq_denom = 1 / ((10e4)** torch.arange(start=0, end=t_dim_channels, step=2, device=self.device) / t_dim_channels)
        even_embeddings = torch.sin(t.repeat(1, t_dim_channels//2)*freq_denom)
        odd_embeddings = torch.cos(t.repeat(1, t_dim_channels//2)*freq_denom)
        return torch.cat([even_embeddings, odd_embeddings], dim=1)


    def forward(self, x, t):
        
        t = t.unsqueeze(1).float() #add singleton dimension for concatenation of time embedings
        t_emb = self.position_embeddings(t, self.t_dim)

        inp = self.input(x) 

        down1 = self.d1(inp,t_emb) 
        down2 = self.d2(down1,t_emb)

        cross_bottom = self.bottom1(down2)
        cross_bottom = self.bottom2(cross_bottom)
        cross_bottom = self.bottom3(cross_bottom)

        up1 = self.u1(cross_bottom, down1, t_emb)
        up2 = self.u2(up1, inp, t_emb)

        output = self.out(up2)
        
        return output
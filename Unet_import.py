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
            
            #all conv blocks will be in channels to in channels except for the last 1-2
            self.convBlocks.append( nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=2, num_channels=in_channels),
            nn.SiLU()
        ))
            
        #append the final convolutional block(s) with out_channels as the output

        #2nd to last block
        # if use_mid_ch:
        #     self.convBlocks.append(nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
        #     nn.GroupNorm(num_groups=2, num_channels=in_channels//2),
        #     nn.SiLU()))

        self.convBlocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=2, num_channels=out_channels),
            nn.SiLU()))

        #accounts for any mismatching channels during addition of residual
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)
        for block in self.convBlocks:
            x = block(x)

        if self.use_residual: return residual + x
        else: return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dimensions=128):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling for downsampling
        self.residualBlock = ResidualBlock(in_channels=in_channels, out_channels=out_channels, use_residual=True)
    

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
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1) #concatenate the cross bottleneck output with current output along their 2nd of 4 dimensions

        x = self.residualBlock(x)
    
        t_embed = self.t_silu(t)
        t_embed = self.t_linear(t_embed)

        t_embed = t_embed[:,:,None,None].repeat(1,1,x.shape[2], x.shape[3])
        
        return x + t_embed
    

class UNet(nn.Module):
    def __init__(self, t_dim=UNet_params["t_dim"], device='cpu'):
        super(UNet, self).__init__()
        
        self.icc = initial_channel_count
        self.scale_rate = UNet_params["scale_rate"]
        
        self.device = device
        self.t_dim = t_dim

        self.input = ResidualBlock(in_channels=self.icc, out_channels=self.icc*self.scale_rate, use_mid_ch=False) #2 to 8

        self.d1 = DownsampleBlock(in_channels=self.icc*self.scale_rate, out_channels=self.icc*self.scale_rate**2, t_dimensions=self.t_dim) #8 to 32 
        self.d2 = DownsampleBlock(in_channels=self.icc*self.scale_rate**2, out_channels=self.icc*self.scale_rate**2, t_dimensions=self.t_dim) #32 to 32

        self.bottom1 = ResidualBlock(in_channels=self.icc*self.scale_rate**2, out_channels=self.icc*self.scale_rate**3, use_residual = False) #32 to 128
        self.bottom2 = ResidualBlock(in_channels=self.icc*self.scale_rate**3, out_channels=self.icc*self.scale_rate**3, use_residual = False) #128 to 128
        self.bottom3 = ResidualBlock(in_channels=self.icc*self.scale_rate**3, out_channels=self.icc*self.scale_rate**2, use_residual = False) #128 to 32

        self.u1 = UpsampleBlock(in_channels=(self.icc*self.scale_rate**2)*2, out_channels=(self.icc*self.scale_rate), t_dimensions=self.t_dim) #64 to 8, not 32 to 8 because channels are concatenated in downsampling
        self.u2 = UpsampleBlock(in_channels=self.icc*self.scale_rate*2, out_channels=(self.icc*self.scale_rate), t_dimensions=self.t_dim) # 16 to 8

        self.out = nn.Conv2d(in_channels=self.icc*self.scale_rate, out_channels=self.icc, kernel_size=1) #8 to 2

    def position_embeddings(self, t, t_dim_channels):
        #Creates a tensor for the frequency denominator of length channels/2 (because step=2 and we want half to be even embeddings and half to be odd embeddings) and normalizes it to the number of total embedding channels
        freq_denom = 1 / ((10e4)** torch.arange(start=0, end=t_dim_channels, step=2, device=self.device) / t_dim_channels)
        even_embeddings = torch.sin(t.repeat(1, t_dim_channels//2)*freq_denom)
        odd_embeddings = torch.cos(t.repeat(1, t_dim_channels//2)*freq_denom)
        return torch.cat([even_embeddings, odd_embeddings], dim=1)



    def forward(self, x, t):
        t = t.unsqueeze(1).float() #add singleton dimension for concatenation of time embedings
        t_emb = self.position_embeddings(t, self.t_dim)

        x = self.input(x) 

        x1 = self.d1(x,t_emb) 
        x2 = self.d2(x1,t_emb)


        x3 = self.bottom1(x2)
        x3 = self.bottom2(x3)
        x3 = self.bottom3(x3)

        x4 = self.u1(x3, x1,t_emb)
        x5 = self.u2(x4, x, t_emb)

        output = self.out(x5)

        return output
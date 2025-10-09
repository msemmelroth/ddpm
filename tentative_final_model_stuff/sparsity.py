import torch

def create_sparsity_mask(inp):
    mask = inp.clone().float()

    # mask[~torch.isfinite(mask)] = -1.0  
    mask[mask>-1] = 1 #here one means on. White pixels are mapped to 1 by default, therefore any pixel <1 is "off" make sure "1 = on" is accounted for in the induce sparsity function
    mask[mask!=1] = -1
    return mask[:,0:1,:,:] 


def separate_sparsity_mask(inp):
    """ we want to go from the input which is the concatneated feature map and sparsity mask 
    along the channel dimension [dim 1] to the map and mask seperated and then element wise multiply them"""
    
    #split the input tensor back into feature map and sparsity mask, assuming that the mask was concatenated to the map along the channel dimension
    channel_cutoff = (inp.shape[1]-1)
    feature_map = inp[:, :channel_cutoff, :, :]
    sparsity_mask = inp[:, channel_cutoff:, : ,:]

    return sparsity_mask, feature_map #(B,1,H,W), (B,C,H,W)

def induce_sparsity(inp):
    """Input: B, C+1, H, W
        Output: B, C, H, W
        
        Gets the mask from the input and maps it to 1s and 0s from what the model predicts
        is a pixel that is dense or a pixel that is unused unused. Mask is then multiplied elment wise with the feature map"""

    mask, map = separate_sparsity_mask(inp)
    
    #1 is white is off -1 is black/gray is on (Initial Picture: White is off, gray/black is function of energy intensity, )

    mask[mask>0] = 1 #approx 1s correspond to an *on* pixel so set mask to 0 to not include this pixel
    mask[mask<=0] = 0 #approx -1s correspond to an off pixel
    return map*mask

# def induce_sparsity_sigmoid(inp): #similar to induce sparsity above but using sigmoid instead of step function
#     map, mask = get_sparsity_mask(inp)
#     return map*torch.sigmoid(mask)
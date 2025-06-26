def create_sparsity_mask(inp):
    """returns a tensor all values of 1 or -1 corresponding to the pixel being on or off"""
    new_inp = inp.clone()  
    new_inp[new_inp<1] = 0
    return (2*new_inp) - 1

def get_sparsity_mask(inp):
    """ we want to go from the input which is the concatneated feature map and sparsity mask 
    along the channel dimension [dim 1] to the map and mask seperated and then element wise multiply them"""
    
    #split the input tensor back into feature map and sparsity mask, assuming that they mask was concatenated to the map
    channel_cutoff = (inp.shape[1])//2
    feature_map = inp[:, :channel_cutoff, :, :]
    sparsity_mask = inp[:, channel_cutoff:, : ,:]

    return sparsity_mask, feature_map

def induce_sparstiy(inp):
    "use during sampling to induce sparsity, inp is the featuremap and concatenated sparsity mask"
    mask, map = get_sparsity_mask(inp)

    #map the sparsity mask to 1s and 0s to use as a sort of filter on the feature map. 
    # mask[mask>0] = 1
    # mask[mask<=0] = 0
    mask = (mask > 0).float()


    return map*mask
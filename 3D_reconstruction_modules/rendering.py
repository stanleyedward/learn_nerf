import torch


def rendering(model, rays_origin, rays_direction, tn, tf, nb_bins = 100, device = 'cpu', white_background = True):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = torch.linspace(tn, tf, nb_bins).to(device) #making the bins #size is no of bins

    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device))) #size is no of bins - 1 so we concat infinity as the last value
    #most nerf papers take the last delta value as infinity
        
    # x = rays_origin + t * rays_direction
    # t.shape = [nb_bins]
    # rays_origin.shape and rays_direction = [nb_rays, nb_bins, 3]
    
    #broadcasting does not work in here therfore reshape the vectors
    
    # t.shape = [1, nb_bins, 1]
    # rays_origin.shape and rays_direction = [nb_rays, 1, 3]
    
    x = rays_origin.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_direction.unsqueeze(1)
    # query the opacity and the colors at each point x
    #query the color and the denisty at each point X to
    # comput the integral
    
    colors, density = model.intersect(x.reshape(-1, 3), rays_direction.expand(x.shape[1], x.shape[0], 3).transpose(0, 1).reshape(-1, 3))
    
    colors = colors.reshape((x.shape[0], nb_bins, 3)) #shape [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], nb_bins, 1)) #[nb_rays, nb_bins, 1]
    
    alpha = 1 - torch.exp(-density.squeeze() * delta.unsqueeze(0)) # shape [nb_rays, nb_bins, 1]
    T = compute_accumulated_transmittance(1 - alpha) # shape [nb_rays, nb_bins, 1]
    weights = T * alpha #[nb_rays, nb_bins]
        
    #since our dataset wasnt consistent with our render as we had a white background
    #if we are in empty space we need to produce value of zero ie black(0) not white (1)
    if white_background:  
        color = (weights.unsqueeze(-1) * colors).sum(1) #shape [nb_rays, 3]
        weight_sum = weights.sum(-1) #[nb_rays] tells if we are in empty space or no via accumulation of denity # when using white background regularization
        
        return color + 1 - weight_sum.unsqueeze(-1)

    else:
        color = (weights.unsqueeze(-1) * colors).sum(1) #shape [nb_rays, 3]
        
    return color

#improved for differentiability
def compute_accumulated_transmittance(betas):
    accumulated_transmittance = torch.cumprod(betas, 1)  
        
    return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device),#since we shift to the right
                      accumulated_transmittance[:, :-1]), dim=1) #sum goes from i =1 to i= N-1


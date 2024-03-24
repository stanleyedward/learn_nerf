import torch


def rendering(model, rays_origin, rays_direction, tn, tf, nb_bins = 100, device = 'cpu'):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = torch.linspace(tn, tf, nb_bins).to(device) #making the bins #size is no of bins

    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10]))) #size is no of bins - 1 so we concat infinity as the last value
    #most nerf papers take the last delta value as infinity
    
    #convert to torch.Tensors
    rays_origin = torch.from_numpy(rays_origin)
    rays_direction = torch.from_numpy(rays_direction)
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
    
    colors, density = model.intersect(x.reshape(-1, 3))
    
    #reshape colors to [nb_rays, nb_bins, 3]
    #reshape density to [nb_rays, nb_bins, 1]
    colors = colors.reshape((x.shape[0], nb_bins, 3)) #shape [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], nb_bins, 1))
    
    alpha = 1 - torch.exp(-density.squeeze() * delta.unsqueeze(0)) # shape [nb_rays, nb_bins, 1]
    T = compute_accumulated_transmittance(1 - alpha) # shape [nb_rays, nb_bins, 1]
    color = (T.unsqueeze(-1) * alpha.unsqueeze(-1) * colors).sum(1) #shape [nb_rays, 3]
    return color


def compute_accumulated_transmittance(betas):
    accumulated_transmittance = torch.cumprod(betas, 1)
    accumulated_transmittance[:, 1:] = accumulated_transmittance[:, :-1] #since we shift to the right
    #sum goes from i =1 to i= N-1
    accumulated_transmittance[:, 0] = 1.  # as T0 should be 1
    #will improve to better differeniability later!
    return accumulated_transmittance


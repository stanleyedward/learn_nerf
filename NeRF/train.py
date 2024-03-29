from tqdm import tqdm
import rendering
import torch


def training(model, optimizer, scheduler, dataloader,tn, tf, nb_bins, nb_epochs, device= 'cpu'):
    
    training_loss = []
    
    progress_bar = tqdm(
    enumerate(dataloader), 
    total=len(dataloader),
    )
    
    for epoch in range(nb_epochs):
        progress_bar.set_description(f"Training Epoch: {epoch}")
        for idx ,batch in progress_bar:
            origin = batch[:, :3].to(device)
            direction = batch[:, 3:6].to(device)
            
            target = batch[:, 6:].to(device)
            
            prediction = rendering.rendering(model, origin, direction, tn, tf, nb_bins, device)
            
            loss = ((prediction - target)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(
            {
                "loss": loss.item() 
            }
            )
            training_loss.append(loss.item())
            
        scheduler.step()
        
        torch.save(model.cpu(), 'models/model_nerf')
        model.to(device)
        
        
    return training_loss
import torch
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def get_rays(datapath, mode='train'):
    print(f"Mode: {mode}")
    pose_file_names = [f for f in os.listdir(datapath + f'/{mode}/pose/') if f.endswith('.txt')]
    intrinsics_file_names= [f for f in os.listdir(datapath + f'/{mode}/intrinsics/') if f.endswith('.txt')]
        
    img_file_names = [f for f in os.listdir(datapath + '/imgs/') if mode in f]
    print(f"Image files length: {len(img_file_names)}")
    print(f"Pose files length: {len(pose_file_names)}")
    print(f"Intrinsic files length: {len(intrinsics_file_names)}")
    
    assert len(pose_file_names) == len(intrinsics_file_names)
    assert len(img_file_names) == len(pose_file_names)
    
    #Read
    N = len(pose_file_names)
    poses = np.zeros((N, 4, 4)) #N 4x4 matrixes (homogeneous)
    intrinsics = np.zeros((N, 4, 4))
    
    images = []
    
    for i in range(N):
        pose_name = pose_file_names[i]
        pose = open(datapath + f'/{mode}/pose/' + pose_name).read().split()
        poses[i] = np.array(pose, dtype=float).reshape(4,4)
        
        # print(poses[i])
        
        intrinsics_name = intrinsics_file_names[i]
        intrinsic = open(datapath + f'/{mode}/intrinsics/' + intrinsics_name).read().split()
        intrinsics[i] = np.array(intrinsic, dtype=float).reshape(4, 4)
        
        # print(intrinsics[i])
        
        #Read images
        image_name = img_file_names[i]
        img = imageio.imread(datapath + '/imgs/' + image_name)
        max_img_intensity = float(img.max()) #255 
        img = img / max_img_intensity #normalizing pixel intensities so theyre between 0-1
        
        images.append(img[None, ...]) #unsqueeze 1st dim in numpy
        
    print(f"Image size: {img.shape}")   
    images = np.concatenate(images)
    print(images.shape)
    
    H = images.shape[1]
    W = images.shape[2]
    
    # remove the 4th dimension to get rid of the alpha channel ie opacity
    if images.shape[3] == 4: #RGBA -> RGB
        images = images[..., :3] * images[..., -1:] + (1-images[..., -1:])
    
    # plt.imshow(images[0])
    # plt.show()
    
    rays_origin = np.zeros((N, H*W, 3))
    rays_direction = np.zeros((N, H*W, 3))
    target_px_values = images.reshape((N, H*W, 3))

    for i in range(N):
        
        camera2world = poses[i]
        f = intrinsics[i,0,0]
        
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)

        dirs = np.stack((u - W / 2,
                        -(v - H / 2),
                        - np.ones_like(u) * f), axis=-1)

        dirs = (camera2world[:3, :3] @ dirs[..., None]).squeeze(-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        
        rays_direction[i] = dirs.reshape(-1, 3)
        rays_origin[i] += camera2world[:3, 3]
        
    return rays_origin, rays_direction, target_px_values
    
    
    
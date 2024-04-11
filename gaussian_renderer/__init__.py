#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, itr=-1, rvq_iter=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    cov3D_precomp = None

    if itr == -1:
        scales = pc._scaling
        rotations = pc._rotation
        opacity = pc._opacity
        
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
        dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        shs = pc.mlp_head(torch.cat([pc._feature, pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
        
    else:
        mask = ((torch.sigmoid(pc._mask) > 0.01).float()- torch.sigmoid(pc._mask)).detach() + torch.sigmoid(pc._mask)
        if rvq_iter:
            scales = pc.vq_scale(pc.get_scaling.unsqueeze(0))[0]
            rotations = pc.vq_rot(pc.get_rotation.unsqueeze(0))[0]
            scales = scales.squeeze()*mask
            rotations = rotations.squeeze()
            opacity = pc.get_opacity*mask

        else:
            scales = pc.get_scaling*mask
            rotations = pc.get_rotation
            opacity = pc.get_opacity*mask
            
        xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
        dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        shs = pc.mlp_head(torch.cat([pc.recolor(xyz), pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D.float(),
        means2D = means2D,
        shs = shs.float(),
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
            }
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
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from icecream import ic
import copy

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spherical_sample_path, generate_spiral_path, generate_spherify_path, gaussian_poses, circular_poses
# import stepfun 



def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


# xy circular 
def render_circular_video(model_path, iteration, views, gaussians, pipeline, background, radius=0.5, n_frames=240): 
    render_path = os.path.join(model_path, 'circular', "ours_{}".format(iteration))
    os.makedirs(render_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    # view = views[0]
    for idx in range(n_frames):
        view = copy.deepcopy(views[13])
        angle = 2 * np.pi * idx / n_frames
        cam = circular_poses(view, radius, angle)
        rendering = render(cam, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def render_video(model_path, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    # render_path_spiral
    # render_path_spherical
    for idx, pose in enumerate(tqdm(generate_ellipse_path(views,n_frames=600), desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def gaussian_render(model_path, iteration, views, gaussians, pipeline, background, args):
    views = views[:10] #take the first 10 views and check gaussian view point 
    render_path = os.path.join(model_path, 'video', "gaussians_{}_std{}".format(iteration, args.std))
    makedirs(render_path, exist_ok=True)

    for i, view in enumerate(views):
        rendering = render(view, gaussians, pipeline, background)["render"]
        sub_path = os.path.join(render_path,"view_"+str(i))
        makedirs(sub_path ,exist_ok=True)
        torchvision.utils.save_image(rendering, os.path.join(sub_path, "gt"+'{0:05d}'.format(i) + ".png"))
        for j in range(10):
            n_view = copy.deepcopy(view)
            g_view = gaussain_poses(n_view, args.mean, args.std)
            rendering = render(g_view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(sub_path, '{0:05d}'.format(j) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, video: bool, circular:bool, radius: float, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        gaussians.precompute()
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        if circular:
            render_circular_video(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,radius)
        # by default generate ellipse path, other options include spiral, circular, or other generate_xxx_path function from utils.pose_utils 
        # Modify trajectory function in render_video's enumerate 
        if video:
            render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        #sample virtual view 
        if args.gaussians:
            gaussian_render(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--circular", action="store_true")
    parser.add_argument("--radius", default=5, type=float)
    parser.add_argument("--gaussians", action="store_true")
    parser.add_argument("--mean", default=0, type=float)
    parser.add_argument("--std", default=0.03, type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.video, args.circular, args.radius, args)

import os
import numpy as np
from tqdm import tqdm
# import uuid
# from argparse import Namespace
# from random import randint

import torch
#from torch.utils.tensorboard import SummaryWriter

from utils.general_utils import safe_state
from utils.utils import str2bool, dump_code, images_to_video

import config.config_blendshapes as config

#import FLAME.transforms as f_transforms
#import FLAME.face_gs_model as f_gaussian_model
import FLAME.face_gs_blend as f_gaussian_model

#import FLAME.mouth_gs_model as mouth_model
from FLAME.dataset import FaceDataset
from FLAME.dataset_dyn import FaceDatasetDyn
from FLAME.dataset_nerfbs import FaceDatasetNerfBS
from FLAME.dataset_ict import FaceDatasetICT
import FLAME.face_renderer as f_renderer

from torchvision import transforms, utils, models
from networks.generator_sup import FaceUNet

#ignore_neck = False
ignore_neck = True
max_displacement_of_blendshape0 = 0.005703532602638
max_displacement_of_blendshape49 = 0.000237277025008

torch.set_num_threads(1)

def mask_function(x,args):
    threshold = max_displacement_of_blendshape49 * 0.1
    L = torch.sqrt(torch.clamp(torch.sum(x * x, dim=1),1e-18,None))
    y = torch.clamp((L-threshold) / (max_displacement_of_blendshape0 - threshold),0,None)
    return y

def config_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Train your network sailor.')
    parser.add_argument('--sh_degree', type=int, default=config.sh_degree, help='sh level total basis is (D+1)*(D+1)')
    parser.add_argument('-s', '--source_path', type=str, default=config.source_path, help='dataset path')
    parser.add_argument('-m', '--model_path', type=str, default=config.model_path, help='model path')
    parser.add_argument("--white_bkgd", type=str2bool, default=config.white_bkgd, help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--data_device", type=str, default=config.data_device)
    parser.add_argument("--reside_image_on_gpu", type=str2bool, default=config.reside_image_on_gpu)
    parser.add_argument("--use_nerfBS", type=str2bool, default=config.use_nerfBS, help='enable to train on NeRFBlendShape dataset')
    parser.add_argument("--use_HR", type=str2bool, default=config.use_HR, help='use high resolution images')
    parser.add_argument("--use_ict", type=str2bool, default=config.use_ict, help='use ICT models')
    
    # optimizer
    parser.add_argument("--iterations", type=int, default=config.iterations)
    parser.add_argument("--position_lr_init", type=float, default=config.position_lr_init)
    parser.add_argument("--position_lr_final", type=float, default=config.position_lr_final)
    parser.add_argument("--position_lr_delay_mult", type=float, default=config.position_lr_delay_mult)
    parser.add_argument("--position_lr_max_steps", type=int, default=config.position_lr_max_steps)

    parser.add_argument("--feature_lr", type=float, default=config.feature_lr)
    parser.add_argument("--opacity_lr", type=float, default=config.opacity_lr)
    parser.add_argument("--scaling_lr", type=float, default=config.scaling_lr)
    parser.add_argument("--rotation_lr", type=float, default=config.rotation_lr)
    parser.add_argument("--percent_dense", type=float, default=config.percent_dense)
    # parser.add_argument("--lambda_dssim", type=float, default=config.lambda_dssim)

    parser.add_argument("--camera_extent", type=float, default=config.camera_extent)
    parser.add_argument("--convert_SHs_python", type=str2bool, default=config.convert_SHs_python)
    parser.add_argument("--compute_cov3D_python", type=str2bool, default=config.compute_cov3D_python)
    parser.add_argument("--debug", type=str2bool, default=False)

    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=list, default=config.test_iterations)
    parser.add_argument("--quiet", action="store_true")
    #parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=config.checkpoint_iterations)

    # face
    parser.add_argument('--flame_geom_path', type=str, default=config.flame_geom_path)
    parser.add_argument('--flame_lmk_path', type=str, default=config.flame_lmk_path)
    parser.add_argument('--back_head_file', type=str, default=config.back_head_file)

    parser.add_argument('--init_face_point_number', type=int, default=config.init_face_point_number)
    parser.add_argument('--num_shape_params', type=int, default=config.num_shape_params)
    parser.add_argument('--num_exp_params',type=int ,default=config.num_exp_params)

    parser.add_argument('--basis_lr_decay', type=float, default=config.basis_lr_decay)
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay)

    # test
    parser.add_argument('--dump_for_viewer', type=str2bool, default=True)
    parser.add_argument('--render_seq', type=str2bool, default=False)
    parser.add_argument('--render_train', type=str2bool, default=False)
    parser.add_argument('--render_test', type=str2bool, default=True)
    parser.add_argument('--put_text', type=str2bool, default=False)
    parser.add_argument('--load_iteration', type=int, default=-1) # Default = search newest

    args, unknown = parser.parse_known_args()

    if len(unknown) != 0:
        print(unknown)
        exit(-1)

    args.flame_template_path = os.path.join(args.source_path, "canonical.obj")

    print("Test " + args.model_path)

    # Initialize system state (RNG)
    #safe_state(args.quiet)

    os.makedirs(args.model_path,exist_ok=True)
    #dump_code(os.path.dirname(os.path.abspath(__file__)), args.model_path)

    return args

def to_image(x):
    if isinstance(x,torch.Tensor):
        x = x.cpu().detach().numpy()
    x = np.clip(np.round(x * 255.),0.,255.)
    return np.asarray(x,dtype=np.uint8)

def render_set(model_path, name, iteration, views, gaussians, args, background):
    from os import makedirs
    import cv2

    if args.use_HR:
        dataset, views = views
        dataset.create_load_seqs(views)

    render_path = os.path.join(model_path, name, "split_{}".format(iteration))
    merge_path = os.path.join(model_path, name, "join_{}".format(iteration))
    render_path_gt = os.path.join(model_path, name, "gt")
    render_path_pred = os.path.join(model_path, name, "pred")
    
    render_path_sup = os.path.join(model_path, name, "sup")
    
    makedirs(render_path, exist_ok=True)
    makedirs(merge_path, exist_ok=True)
    
    makedirs(render_path_gt, exist_ok=True)
    makedirs(render_path_pred, exist_ok=True)
    
    makedirs(render_path_sup, exist_ok=True)

    #render_path_gt = os.path.join(model_path, name, "", "{}".format(iteration))
     
    face_gaussians = gaussians[0]
    
    args.input_size = 512
    args.output_size = 512
    args.channel_multiplier = 2
    
    g_ema = FaceUNet(args.input_size, args.output_size, args.channel_multiplier).to("cuda")
    g_ema.eval()
    
    args.ckpt = os.path.join(model_path, "facenet_700000.pt")
    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt)
    g_ema.load_state_dict(ckpt["g_ema"], strict=True)
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    
    blend_lst = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if args.use_HR:
            view = dataset.getData(view, load_mode = 'load')
        face_gaussians.prepare_xyz(view, args) #view.blend[0, 52]
        
        blend_lst.append(view.blend[0,:52].detach().cpu().numpy())

        rendering = f_renderer.render_alpha(view, gaussians, args, background, override_color= None)
        if args.use_nerfBS:
            gt = view.original_image[:, :, 0:3]
            gt = to_image(gt)
            bkg = view.bkg.cuda()
            bkg = bkg.permute(2,0,1)
            image = rendering['render']
            alpha0 = rendering['alpha0']
            image = image + (1-alpha0) * bkg # image with background
            image = image.permute(1,2,0)
            image = to_image(image)
        else:
            gt = view.original_image[:, :, 0:3] * view.mask[:, :, None]
            gt = to_image(gt)
            image = rendering['render'].permute(1,2,0)
            image = to_image(image)
        # vis = face_gaussians.vis(view, args, [f_gaussian_model.View.SHAPE])
        # shape = to_image(vis[0])
        sup_img = g_ema(transform(rendering['render'].unsqueeze(0)))[0]
        sup_img = (sup_img / 2 + 0.5).permute(1, 2, 0)
        image_sup = to_image(sup_img)
        
        # print(171)
        # from IPython import embed 
        # embed()
        
        gt_mask = torch.stack([view.mask, torch.zeros_like(view.mask), torch.zeros_like(view.mask)], dim=-1)
        gt_mask = to_image(gt_mask)

        alpha0 = rendering['alpha0']
        alpha_image = torch.stack([alpha0, torch.zeros_like(alpha0), torch.zeros_like(alpha0)],dim=-1)
        alpha_image = to_image(alpha_image)

        cv2.imwrite(os.path.join(render_path, "gt_%05d.png" % idx),gt[...,::-1])
        cv2.imwrite(os.path.join(render_path, "pred_%05d.png" % idx),image[...,::-1])
        
        cv2.imwrite(os.path.join(render_path_gt, "%05d.png" % idx),gt[...,::-1])
        cv2.imwrite(os.path.join(render_path_pred, "%05d.png" % idx),image[...,::-1])
        cv2.imwrite(os.path.join(render_path_sup, "%05d.png" % idx),image_sup[...,::-1])    
        
        #cv2.imwrite(os.path.join(render_path,"mesh_%05d.png" % idx), shape[...,::-1])

        cv2.imwrite(os.path.join(render_path,"gt_mask_%05d.png" % idx), gt_mask[...,::-1])
        cv2.imwrite(os.path.join(render_path,"pred_mask_%05d.png" % idx), alpha_image[...,::-1])

        # print(178)
        # from IPython import embed 
        # embed()
        # print("Image Shape:", image.shape)
        
        m = np.concatenate([gt,image, image_sup],axis=1)
        if args.render_seq and args.put_text:
            if args.n_extract_ratio == -1:
                tag = "Test" if (idx + args.n_seg >= len(views)) else "Train"
            else:
                tag = "Test" if ((idx // args.n_seg) % args.n_extract_ratio == args.n_extract_ratio - 1) else "Train"
            cv2.putText(m, tag, (m.shape[1] - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(merge_path, "%05d.png" % idx), m[...,::-1])

    images_to_video(merge_path)
    
    blend_np = np.array(blend_lst)
    # print('Shape blend!', blend_np.shape)
    # np.save('blend.npy', blend_np)
    array_list = blend_np.tolist()
    jsonfile = os.path.join(model_path, name, name + '_blend.json')
    json_camera = os.path.join(model_path, name, name + '_camera.json')
    
    import json 
    with open(jsonfile, 'w') as json_file:
        json.dump(array_list, json_file)
        
    # camera = {}
    # camera['id'] = 0
    # camera['img_name'] = '00001'
    # camera['width'] = 512
    # camera['height'] = 512
    
    # camera['position'] = 
    # camera['rotation'] = 
    # camera['fy'] = 237
    # camera['fx'] = 237
     
    # with open(json_camera, 'w') as json_file:
    #     json.dump(camera, json_file)
    
    # print(192)
    # from IPython import embed 
    # embed()

def searchForMaxIteration2(folder):
    saved_iters = []
    for fname in os.listdir(folder):
        file, ext = os.path.splitext(fname)
        if ext == '.pth':
            start_id = fname.rfind('chkpnt') + 6
            saved_iters.append(int(file[start_id:]))
    print(saved_iters)
    return max(saved_iters)

def render_sets(args, render_train : bool, render_test : bool, render_seq:bool):
    with torch.no_grad():

        if args.use_nerfBS:
            dataset = FaceDatasetNerfBS(args.source_path, shuffle=False)
        elif args.use_ict:
            dataset = FaceDatasetICT(args.source_path, shuffle=False)
            dataset.prepare_data(reside_image_on_gpu=args.reside_image_on_gpu,device=args.data_device)
        else:
            if args.use_HR:
                dataset = FaceDatasetDyn(args.source_path, shuffle=False, ratio=2.0)
            else:
                dataset = FaceDataset(args.source_path, shuffle=False)
        dataset.prepare_data(reside_image_on_gpu=args.reside_image_on_gpu, device=args.data_device)
        dummy_frame = dataset.output_list[0]

        args.n_seg = dataset.n_seg
        args.n_extract_ratio = dataset.n_extract_ratio

        face_gaussians = f_gaussian_model.GaussianModel(args.sh_degree)
        face_gaussians.create_from_face(dummy_frame, args, args.camera_extent)


        if args.load_iteration is None or args.load_iteration == -1:
            load_iteration = searchForMaxIteration2(args.model_path)
        else:
            load_iteration = args.load_iteration

        fix_checkpoint = os.path.join(args.model_path,"fix_chkpnt" + str(load_iteration) + ".pth")
        (model_params, first_iter) = torch.load(fix_checkpoint)
        face_gaussians.restore(model_params, args)

        #f_transforms.rigid_transfer(dataset, face_gaussians, args, gen_local_frame=False)
        
        ## Generate blendshape consistency scalar
        #f_transforms.get_expr_consistency_face(face_gaussians, dummy_frame, mask_function, args, ignore_neck=ignore_neck)
        ## Generate deformation transfers for each expression blendshape
       # f_transforms.get_expr_rot(face_gaussians, dummy_frame, args, light=True)
        ## Get pose blendshapes and eyelid blendshapes
        #f_transforms.get_pose_tensor(face_gaussians, args)
        ## Get joints and joint transfers for each frame.
        #f_transforms.from_mesh_to_point(dataset, face_gaussians, args)
        ## Generate jaw transfer
        # used to transfer lower teeth


        if args.dump_for_viewer: # Dump files for C++/CUDA viewer
            fix_checkpoint_path = os.path.splitext(fix_checkpoint)[0]
            #face_gaussians.save_npy_forviewer(fix_checkpoint_path)
            print('dump npy %s' % fix_checkpoint_path)

        gaussians = [face_gaussians]

        bg_color = [1,1,1] if args.white_bkgd else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_ids = dataset.getTrainCameras()
        test_ids = dataset.getTestCameras()
        if args.use_HR:
            train_dataset = (dataset, train_ids)
            test_dataset = (dataset, test_ids)
            seq_dataset = (dataset, train_ids + test_ids)
        else:
            train_dataset = [dataset.output_list[i] for i in train_ids]
            test_dataset = [dataset.output_list[i] for i in test_ids]
            seq_dataset = dataset.output_list

        if render_seq:
            render_set(args.model_path, "seq", load_iteration, seq_dataset, gaussians, args, background)
        if render_train:
            render_set(args.model_path, "train", load_iteration, train_dataset, gaussians, args, background)
        if render_test:
            render_set(args.model_path, "test", load_iteration, test_dataset, gaussians, args, background)

def test():

    args = config_parse()
    #safe_state(args.quiet)
    render_sets(args, args.render_train, args.render_test, args.render_seq)

if __name__ == "__main__":
    test()
#python test_sup.py --source_path id2_ori --model_path ./output/id2 --use_ict True --render_train True --load_iteration 80000
# You can use this to convert a .ply file to a .splat file programmatically in python
# Alternatively you can drag and drop a .ply file into the viewer at https://antimatter15.com/splat

from plyfile import PlyData
import numpy as np
import argparse
from io import BytesIO
import glob
import os

def process_ply_to_splat(ply_file_path):
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x_0"], v["y_0"], v["z_0"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        
        # color = np.array(
        #     [
        #         v["RGB_0"],
        #         v['RGB_1'],
        #         v['RGB_2'],
        #         1 / (1 + np.exp(-v["opacity"])),
        #     ]cd 
        # )
        
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )
        
        for i in range(1, 51+1):
            x_name = 'x_{}'.format(i)
            y_name = 'y_{}'.format(i)
            z_name = 'z_{}'.format(i)
            position_temp = np.array([v[x_name], v[y_name], v[z_name]], dtype=np.float32)
            buffer.write(position_temp.tobytes())

    return buffer.getvalue()
def save_splat_file(splat_data, output_path):
    with open(output_path, "wb") as f:
        f.write(splat_data)
def main():
    parser = argparse.ArgumentParser(description="Convert PLY files to SPLAT format.")

    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="ply_dir", 
        help="The directory containing the input files.")
    parser.add_argument(
        "--output", "-o", default="output.splat", help="The output SPLAT file."
    )
    args = parser.parse_args()
    if args.input_dir:
        input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")))
    else:
        print('Input Dir is none ')
        
    for input_file in input_files:
        print(f"Processing {input_file}...")
        splat_data = process_ply_to_splat(input_file)
        output_file = (
            args.output if len(input_files) == 1 else input_file + ".splat"
        )
        save_splat_file(splat_data, output_file)
        print(f"Saved {output_file}")
if __name__ == "__main__":
    main()
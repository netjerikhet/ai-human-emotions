import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from batchnorm_sync import DataParallelWithCallback

from model.generator import OcclusionAwareGenerator
from model.keypoint_detector import KPDetector, normalize_kp
from scipy.spatial import ConvexHull


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.safe_load(f)
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()
    return generator, kp_detector

    def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                source = source.cuda()
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                    use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config(.yaml file)")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path of the checkpoint to be loaded into the model")
    parser.add_argument("--emotion", default='neutral', help="emotion to be applied to the source image")
    parser.add_argument("--source_image", default='', help="path to source image to be animated")
    parser.add_argument("--driving_video", default='', help="path to the driving video which will be used to animate the source image")
    parser.add_argument("--result_video", default='./output_video/result.mp4', help="path to output")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="set to True if you want to use cpu for training")
    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    if os.path.exists('./output_video/'):
        pass
    else:
        os.mkdir('./output_video/')
    opt = parser.parse_args()
    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        print("Error loading the video please check file  path")
        pass
    reader.close()
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


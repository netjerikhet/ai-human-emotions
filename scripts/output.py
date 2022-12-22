import matplotlib
matplotlib.use('Agg')
import os, sys
import random
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from face_cropper import FaceCropper
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from batchnorm_sync import DataParallelWithCallback
import cv2
from model.generator import OcclusionAwareGenerator
from model.keypoint_detector import KPDetector, normalize_kp
from scipy.spatial import ConvexHull
import wandb
# from super_resolution import test
from metrics.ssim_check import metric_evaluation


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.safe_load(f)
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    # if not cpu:
    #     generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    # if not cpu:
    #     kp_detector.cuda()
    
    # if cpu:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # else:
    #     checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()   
    return generator, kp_detector

# def change_white_background(image):
#     hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # rgb to hsv color space
#     s_ch = hsv_img[:, :, 1]  # Get the saturation channel
#     cv2_imshow(s_ch)
#     thesh = cv2.threshold(s_ch, 10, 255, cv2.THRESH_BINARY)[1]  # Apply threshold - pixels above 5 are going to be 255, other are zeros.
#     thesh = cv2.morphologyEx(thesh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))  # Apply opening morphological operation for removing artifacts.
#     print(thesh)
#     cv2.floodFill(thesh, None, seedPoint=(0, 0), newVal=150, loDiff=4, upDiff=4)  # Fill the background in thesh with the value 128 (pixel in the foreground stays 0.

#     image[thesh == 150] = (140, 230, 255)
#     return image

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

def get_random_folders(folder_list):
    return random.sample(folder_list, 5)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config(.yaml file)")
    parser.add_argument("--video_dir", default="data/test_cropped_videos/source_videos/")
    parser.add_argument("--checkpoint", required=True, default='vox-cpk.pth.tar', help="path of the checkpoint to be loaded into the model")
    parser.add_argument("--emotion", default='neutral', help="emotion to be applied to the source image")
    parser.add_argument("--source_image", default='', help="path to source image to be animated")
    parser.add_argument("--driving_video", default='', help="path to the driving video which will be used to animate the source image")
    parser.add_argument("--result_video", default='./output_video/result.mp4', help="path to output")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="set to True if you want to use cpu for training")
    parser.add_argument("--save", default=False, help="To save output videos")
    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    if os.path.exists('./output_video/'):
        pass
    else:
        os.mkdir('./output_video/') 
    opt = parser.parse_args()
    detector = FaceCropper()
    metric_eval = metric_evaluation()
    wandb_obj = wandb.init(project="ouput_videos_of_model", entity="ai-human-emotion")
    test_image_list = os.listdir("data/test_images/")
    test_path = "data/test_images/"
    video_dir = opt.video_dir
    source_videos_list = os.listdir(video_dir)
    emotions_list = os.listdir(f"{video_dir}{source_videos_list[0]}/light_uniform/")
    videos = []
    for source_video in get_random_folders(source_videos_list):
        for emotion in emotions_list:
            video_path = f"{video_dir}{source_video}/light_uniform/{emotion}/camera_front/"
            print(video_path)
            if os.path.exists(video_path):
                video_name = os.listdir(video_path)[0]
                video_path = video_path + video_name
                videos.append(video_path)

    for video_path in videos:
        for test_image in test_image_list:
            source_image = imageio.imread(test_path+test_image)
            faces = detector.detect_face(source_image, show_result=False)
            source_image = detector.generate_cropped_source_image(faces, source_image, save_picture=False)
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

            if len(faces) == 0:
                print("Face not recognized!!")
                exit()
            reader = imageio.get_reader(video_path, mode='I', format='FFMPEG')
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

            print("source Image length", source_image.shape)
            #source_image = change_white_background(source_image)
            driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
            generator, kp_detector = load_checkpoints(config_path=opt.config, 
            checkpoint_path=opt.checkpoint, cpu=opt.cpu)
            predictions = make_animation(source_image, driving_video, generator, 
            kp_detector, cpu=opt.cpu)
            imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=30)
            reader = imageio.get_reader(opt.result_video, mode='I', format='FFMPEG')
            predictions = []
            predictions = [frame for frame in reader]
            # hq_predictions = test.sr_video_forward(predictions)
            driving_video = [img_as_ubyte(frame) for frame in driving_video]
            if hq_predictions is not None:
                imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in hq_predictions], fps=30)
                score = metric_eval.ssim_psnr_score(driving_video, hq_predictions)
            else:
                score = metric_eval.ssim_psnr_score(driving_video, predictions)
            ssim_score, psnr_score = score[0], score[1]
            wandb_obj.log(
                    {f"Generated video vs driving video-{video_path}": 
                                [wandb.Video(opt.result_video, fps=30, format="mp4", 
                                caption=f"SSIM score:{ssim_score}, PSNR score: {psnr_score}"),
                                     wandb.Video(video_path, fps=30, format="mp4")]})
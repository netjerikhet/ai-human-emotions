from skimage import metrics
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np

ssim_score_total = 0
psnr_score_total = 0
driving_video_path = "data/test_cropped_videos/source_videos/M011/light_uniform/surprise/camera_front/M011_light_uniform_surprise_camera_front.mp4"
output_path = "data/resultant_videos/M011_light_uniform_surprise_camera_front.mp4"
reader = imageio.get_reader(driving_video_path, mode='I', format='FFMPEG')
driving_video = [frame for frame in reader]
reader = imageio.get_reader(output_path, mode='I', format='FFMPEG')
output_video = [frame for frame in reader]

for driving_frame, output_frame in zip(driving_video, output_video):
    resized_driving_frame = cv2.resize(driving_frame, (256, 256), interpolation = cv2.INTER_AREA)
    resized_output_frame = cv2.resize(output_frame, (256, 256), interpolation = cv2.INTER_AREA)
    
    # Convert the images to grayscale
    resized_driving_frame = cv2.cvtColor(resized_driving_frame, cv2.COLOR_BGR2GRAY)
    resized_output_frame = cv2.cvtColor(resized_output_frame, cv2.COLOR_BGR2GRAY)
    
    (ssim_score, diff) = metrics.structural_similarity(resized_driving_frame, resized_output_frame, full=True)
    ssim_score_total = ssim_score_total + ssim_score

    psnr_score = metrics.peak_signal_noise_ratio(resized_driving_frame, resized_output_frame)         
    psnr_score_total = psnr_score_total + psnr_score
print(len(driving_video))
print(ssim_score_total/len(driving_video))
print(psnr_score_total/len(driving_video))
# print("ssim score list:", ssim_score_list)
# print("psnr score list:", psnr_score_list)
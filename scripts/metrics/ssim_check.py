from skimage import metrics
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np

class metric_evaluation:
    def __init__(self):
        """
        This class contains functions for metric evaluation of driving video and generated output video

        """
        
    def ssim_psnr_score(self, driving_video, output_video):
        """
            Function to calculate the SSIM and PSNR score using metrics.structural_similarity and metrics.peak_signal_noise_ratio
        """
        ssim_score_total = psnr_score_total = 0
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

        mean_ssim_score = ssim_score_total/len(driving_video)
        mean_psnr_score = psnr_score_total/len(driving_video)
        return [mean_ssim_score, mean_psnr_score]
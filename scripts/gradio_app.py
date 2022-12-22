import gradio as gr
import imageio
import os, re, os.path
import skimage
from skimage import img_as_ubyte
import numpy
from output import load_checkpoints, make_animation
from metrics.ssim_check import metric_evaluation
# from super_resolution import test
import cv2
from skimage import metrics

metric_eval = metric_evaluation()
default_checkpoint_path="data/pretrained_model/vox-cpk.pth.tar"
dir = 'data/test_cropped_videos/source_videos/'

def perform_animation_and_get_score(selected_image, gender, emotion):

    checkpoint_file = "data/pretrained_model/best-checkpoint-{}.pth.tar".format(emotion)
    generator, kp_detector = load_checkpoints(config_path='code/config/deeper_forensics.yaml', 
                            checkpoint_path = checkpoint_file if os.path.isfile(checkpoint_file) else default_checkpoint_path)
    score_info = {}
    if gender == "Men":
        videos_list = [dir+'{}/light_uniform/{}/camera_front/{}_light_uniform_{}_camera_front.mp4'.format(x, emotion, x, emotion) 
        for x in list(filter(re.compile("M.*").match, os.listdir(dir)))]
    else:
        videos_list = [dir+'{}/light_uniform/{}/camera_front/{}_light_uniform_{}_camera_front.mp4'.format(x, emotion, x, emotion) 
        for x in list(filter(re.compile("W.*").match, os.listdir(dir)))]
    # for i in videos_list:
        # os.path.isfile(i)  #check if file exists
    ssim_list = {}
    for video in videos_list:
        if os.path.exists(video):
            reader = imageio.get_reader(video, mode='I', format='FFMPEG')
            _, video_name = os.path.split(video)
            fps = reader.get_meta_data()['fps']
            driving_video_frame = []
            for frame in reader:
                driving_video_frame.append(frame)
                break
            resized_driving_frame = cv2.resize(driving_video_frame[0], (256, 256), interpolation = cv2.INTER_AREA)
            resized_source_image = cv2.resize(selected_image, (256, 256), interpolation = cv2.INTER_AREA)
            # Convert the images to grayscale
            resized_driving_frame = cv2.cvtColor(resized_driving_frame, cv2.COLOR_BGR2GRAY)
            resized_source_image = cv2.cvtColor(resized_source_image, cv2.COLOR_BGR2GRAY)
            (ssim_score, diff) = metrics.structural_similarity(resized_driving_frame, resized_source_image, full=True)
            ssim_list[video] = ssim_score
    ssim_list = sorted(ssim_list.items(), key=lambda x: x[1], reverse=True)
    best_video_list = []
    i = 0
    print("10 best driving videos scores")
    for key, value in ssim_list:
        best_video_list.append(key)
        _, name = os.path.split(key)
        print(f"{name}: {value}")
        i+=1
        if i == 10:
            break

    for video in best_video_list:
        reader = imageio.get_reader(video, mode='I', format='FFMPEG')
        _, video_name = os.path.split(video)
        fps = reader.get_meta_data()['fps']
        driving_video = predicted_video = []
        for frame in reader:
            driving_video.append(frame)
        predictions = make_animation(
            skimage.transform.resize(numpy.asarray(selected_image), (256, 256)),
            [skimage.transform.resize(frame, (256, 256)) for frame in driving_video],
            generator,
            kp_detector,
            relative=True,
            adapt_movement_scale=True
        )
        predicted_video = [img_as_ubyte (frame) for frame in predictions]
    # try:
    #     hq_predictions = test.sr_video_forward(predicted_video)
    #     imageio.mimsave('data/resultant_videos/{}'.format(video_name), hq_predictions, fps=fps)
    #     score_info[video_name] = metric_eval.ssim_psnr_score(driving_video, hq_predictions)
    # except:
    imageio.mimsave('data/resultant_videos/{}'.format(video_name), predicted_video, fps=fps)
    score_info[video_name] = metric_eval.ssim_psnr_score(driving_video, predicted_video)
    print(score_info)
    return score_info
    #     hq_predictions = test.sr_video_forward(predicted_video)
    #     if hq_predictions is not None:
    #         imageio.mimsave('data/resultant_videos/{}'.format(video_name), hq_predictions, fps=fps)
    #         score_info[video_name] = metric_eval.ssim_psnr_score(driving_video, hq_predictions)
    #     else:
    #         imageio.mimsave('data/resultant_videos/{}'.format(video_name), predicted_video, fps=fps)
    #         score_info[video_name] = metric_eval.ssim_psnr_score(driving_video, predicted_video)
    # # print(score_info)
    # return score_info


def generate_output(selected_image, gender, emotion, score):
    metric_info = perform_animation_and_get_score(selected_image, gender, emotion)
    if score == 'ssim':
        score_info = {video: score[0] for video, score in metric_info.items()}
    if score == 'psnr':
        score_info = {video: score[1] for video, score in metric_info.items()}

    print('all scores', metric_info)
    # top_2_scores = sorted(score_info.values(), reverse=True)[:2]
    largest_score_video = next(iter(score_info))
    largest_score = score_info[next(iter(score_info))]

    best_generated_video = 'data/resultant_videos/{}'.format(largest_score_video)
    sr_video = []
    reader = imageio.get_reader(best_generated_video, mode='I', format='FFMPEG')
    for frame in reader:
        sr_video.append(frame)
    # hq_predictions = test.sr_video_forward(sr_video)
    # imageio.mimsave('data/resultant_videos/sr_{}'.format(largest_score_video), hq_predictions, fps=30)
    # best_generated_sr_video = 'data/resultant_videos/sr_{}'.format(largest_score_video)
    # best_driving_video = dir+'{}/light_uniform/{}/camera_front/{}'.format(largest_video_score_name.rsplit("_", 5)[0], emotion, largest_video_score_name)
    return best_generated_video, largest_score_video, largest_score


image = gr.inputs.Image ( shape=(256, 256), label="Input source Image" )
gender = gr.inputs.Dropdown(["Men", "Women"], label="Input preferred gender")
emotions = gr.inputs.Dropdown ( ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                             label="Input Emotion for driving video" )
metric = gr.inputs.Dropdown ( ["ssim", "psnr"], label="Input score evaluation metric" )

top_scores = gr.outputs.Textbox (label = "Score for best video")
top_videos = gr.outputs.Textbox (label = "Best driving video")

css = "#video1 {max-width: 500px; max-height: 500px; display: inline-block;}"


gr.Interface(fn=generate_output, css=css,
            inputs=[image, gender, emotions, metric],
            outputs=[gr.Video(elem_id='video1'), top_videos, top_scores], 
            interpretation='default').launch(debug='False', share=True)

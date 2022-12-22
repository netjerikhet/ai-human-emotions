import os

video_path = "./data/test_cropped_videos/source_videos"
source_videos_list = os.listdir(video_path)
emotions_list = os.listdir(f"{video_path}/{source_videos_list[0]}/light_uniform/")
print(emotions_list)
for source_video in source_videos_list:
    for emotion in emotions_list:
        video_dir = f"{video_path}/{source_video}/light_uniform/{emotion}/camera_front/"
        if os.path.exists(video_dir):
            if len(os.listdir(video_dir)) == 0:
                print("deleting:", video_dir)
                os.rmdir(video_dir)
        else:
            print("Missing directory:", video_dir)
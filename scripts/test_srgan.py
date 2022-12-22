from super_resolution import test
import imageio
import cv2

from skimage import img_as_ubyte

video_path = "data/test_cropped_videos/source_videos/M005/light_uniform/angry/camera_front/M005_light_uniform_angry_camera_front.mp4"
#image_path = "data/test_images/face_6.png"
#img = imageio.imread(image_path)
reader = imageio.get_reader(video_path)
video = []
for frame in reader:
    video.append(frame)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#print(img)
hq = test.sr_video_forward(video)
#print("sdalkjfhasdlkfjhsdfklfhsadlkf",hq[0])
#imageio.imwrite("output_video/hq.jpg", hq[0])
imageio.mimsave("output_video/hq.mp4", [img_as_ubyte(frame) for frame in hq], fps=30)
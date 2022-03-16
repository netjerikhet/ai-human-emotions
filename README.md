# ai-human-emotions
Use of Deep Learning algorithms to add high quality motion to still photos

About SIT

SIT is an international institute founded by entrepreneurs, led by scientists and advanced by world-class researchers.

Founded in 2019 and headquartered in Schaffhausen, Switzerland, SIT’s contemporary design creates a unique ecosystem where the world’s leading experts in Computer Science, Physics, and Business come together to find innovative solutions to global challenges through transformative technological advances. 

SIT is comprised of an educational institute, research center and vast internal and external ecosystem, collaborating with some of the world's top academic and science institutions including Carnegie-Mellon University and the School of Computing at the National University of Singapore (NUS).

Project Context

A still photo captures a single moment in time and represents 2D image of real-world 3D scene. A lot of information from original 3D scene is lost. Best pictures from professional photographers use many technics applies to composition, lighting, focal length, shutter speed, focus settings and such to preserve and emphasize real scene dynamic and 3D information and allow viewer to reconstruct it in his mind looking at static 2D image.  

Human mind keeps whole history of its visual interactions with the real world and has ability to reconstruct dynamic 3D scene from static 2D image in his imagination. 

Using deep learning algorithms computer program (same as human mind) could generate 2-10 seconds live video from still photo that would look like a realistic high-quality video.

The resulted prototype solution can be used in various entertainment products – web app and/or mobile app which can create short video from any still photo with people uploaded by user.  

Project Description – Human Emotions

Create prototype solution using deep learning algorithms which do the following: 

-	take still photo in JPG format as input
-	detect people faces and facial features – nose, mouth, ears etc.
-	choose one of the human emotions from the list – joy, fear, anger, sadness, disgust, surprise
-	generate 2-10 seconds video in MPEG4 format
-	in this video animate detected faces and facial features to show dynamic expression of chosen emotion
-	resulted video should be realistic enough for uniformed person to perceive it as real-life video 
-	there should be some random component in animation to avoid synchronized movement of different faces in the same video, which will look unnatural
-	to ensure smoothness of the motion resulted video should be rendered with 60 frames per second  

In scope of the project:

-	using Cascaded Convolutional Networks perform faces and facial features detection on the photo
-	using Generative Adversarial Networks perform generate number (60 * video length in seconds) of photos with changes in faces and facial features representing chosen emotion like joy, fear, anger, sadness, disgust, surprise or any other
-	animation should contain random component to achieve 2 goals – 1-st animation from run to run should be a little different – like real person cannot move exactly the same two times in a row, 2-nd – different people on the same photo should move slightly different to avoid “puppet effect”
-	nearby people’s parts like necks, hair, shoulders had to be animated as well to achieve consistency of the motion
-	advanced - using Generative Adversarial Networks background behind animated parts should be reconstructed to avoid undesirable artefacts on the edges of animated parts
-	advanced – if during animation (for example small rotation) hidden people parts become visible – these parts should be reconstructed using Generative Adversarial Networks
-	advanced – if there was a visible lighting on the animated parts (which is always the case for 3D looking photos) – such as highlights, shadows – during animation these lighting artifacts should animate consistently with lighting sources – sun, lamps (which are static in this animated video and don’t move when people rotate heads)

It is possible to simplify requirements to achieve better results in a reasonable time – for example choose portrait photos with single person, captured on homogenous background using flat light.

The project should be carried out the following work:

-	Research and choosing pretrained ML models to perform some of the steps described above
-	Creation and/or modification ML models to perform some of the steps described above
-	Training models using hardware provided by SIT
-	Creation of a prototype solution capable of performing the functions required above
-	Testing and presentation of developed prototype

Product should be written on Python with using of popular ML libraries like TensorFlow or PyTorch. 

During this project it is allowed to use any open sourced 3-rd party components - ML models, libraries, code under licenses which allows usage in commercial projects (such a MIT license, Apache license and such).

It is important to find and use appropriate datasets for models training. One of the good examples of relevant faces datasets is celebA - https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.

This is a research project, and we assume that project timing may change significantly depend on the research results. There is a possibility that some of the project goals and phases won’t be achieved during first year and in that case, we are going to continue this project on the next year with the same or new team. Keeping this in mind we consider following project timeline:
 
-	Phase 1 – research Cascaded Convolutional Networks for faces and facial features detection, choosing 3-rd party pre-trained model or train model yourself
-	Phase 2 – research Generative Adversarial Networks for generating frames of animated faces, find and select datasets for model training
-	Phase 3  - create Generative Adversarial Network for one or number of selected emotions and do model training 
-	Phase 4 - creation prototype using trained model from Phase 3
-	Phase 5 – fine tune prototype and fix some of the issues found in Phase 4

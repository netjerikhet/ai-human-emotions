# Class to detect face and crop it from the driving video and source image

import numpy as np
import cv2
class FaceCropper(object):

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, image, show_result):
        img = np.asarray(image)
        if (img is None):
            print("Can't open image file")
            return 0

        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return 0

        if (show_result):
            i = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for (x, y, w, h) in faces:
                cv2.rectangle(i, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.imshow(i)

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]
        return faces

    def generate_cropped_face(self, faces, image, save_picture):
        img = np.asarray(image)
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r) 
            ny = int(centery - r)
            nr = int(r * 2) 
            faceimg = img[ny-120:ny+nr+120, nx-150:nx+nr+150]
            lastimg = cv2.resize(faceimg, (256, 256))
            lastimg = cv2.cvtColor(lastimg, cv2.COLOR_RGB2BGR)
            #cv2.imshow('', lastimg)
            if save_picture:
              cv2.imwrite("cropped_image.png", cv2.cvtColor(lastimg, cv2.COLOR_RGB2BGR))
            return lastimg

    def generate_cropped_source_image(self, faces, image, save_picture):
        img = np.asarray(image)
        padding = 50
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r) 
            ny = int(centery - r)
            nr = int(r * 2) 
            faceimg = img[ny-padding:ny+nr+padding, nx-padding:nx+nr+padding]
            lastimg = cv2.resize(faceimg, (256, 256))
            lastimg = cv2.cvtColor(lastimg, cv2.COLOR_RGB2BGR)
            #cv2.imshow('', lastimg)
            if save_picture:
              cv2.imwrite("cropped_image.png", cv2.cvtColor(lastimg, cv2.COLOR_RGB2BGR))
            return lastimg
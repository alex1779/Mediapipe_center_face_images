# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:15:23 2022

@author: Ale
"""

import argparse
import cv2
import mediapipe as mp
import imutils
from data.centerface import frontalFace, save_image, fix_eye


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='data/example1.jpg',
                    help='Please specify path for image', required=True)
opt = parser.parse_args()


fixEye = False
saved = False
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                           refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
    while True:
        image = cv2.imread(opt.img_path)
        
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:

            annotated_image = frontalFace(
                image=annotated_image,
                landmark_list=face_landmarks,
            )

        image_aux = annotated_image.copy()

        if fixEye is not False:
            annotated_image = fix_eye(image_aux)

        if saved is not False:
            save_image(annotated_image)
            saved = False
            cv2.putText(annotated_image, "Image saved!", (20, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.putText(annotated_image, "Press Esc to quit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(annotated_image, "Press S to save", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        cv2.putText(annotated_image, "Press F to fix eye", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        annotated_image = imutils.resize(annotated_image, height=600)
        
        cv2.imshow('Original', image)
        cv2.imshow('Centered image', annotated_image)
        keypressed = cv2.waitKey(0)

        if keypressed == 27:
            cv2.destroyAllWindows()
            break

        if keypressed == ord('s'):
            print('Image Saved')
            saved = True

        if keypressed == ord('f'):
            fixEye = True

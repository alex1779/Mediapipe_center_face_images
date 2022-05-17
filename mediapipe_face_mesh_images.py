# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:15:23 2022

@author: Ale
"""

import cv2
import mediapipe as mp
import imutils
from drawing_orig import drawLandmarkPoints, drawTesselation, drawContourFace, drawNew, cannonicalFace


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# path_in = 'data/brad pitt.jpg'
# path_in = 'data/angelina jolie.jpg'
# path_in = 'data/angelina jolie 2.jpg'

path_in = 'data/109.6.jpg'

path_cannon = 'data/cannon11.jpg'
# # For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

    image = cv2.imread(path_in)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()
    annotated_image = imutils.resize(annotated_image, height=720)
    for face_landmarks in results.multi_face_landmarks:

        # drawLandmarkPoints(
        #     image=annotated_image,
        #     face_landmarks=face_landmarks,
        #     color=(255, 0, 0),
        #     radious=2
        #     )

        # drawTesselation(
        #     image=annotated_image,
        #     face_landmarks=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
        #     color=(0, 0, 255),
        #     thickness=1
        #     )

        # drawContourFace(
        #     image=annotated_image,
        #     face_landmarks=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_FACE_OVAL,
        #     color=(0, 255, 0),
        #     thickness=2
        #     )

        drawNew(
            image=annotated_image,
            face_landmarks=face_landmarks,
            connections=mp_face_mesh.FACEMESH_FACE_OVAL,
            color=(0, 255, 0),
            thickness=2
        )

        cannonicalFace(
            image=annotated_image,
            path_cannon=path_cannon

        )
    exit()
    # annotated_image = imutils.resize(annotated_image, height=720)
    # cv2.imshow('MediaPipe Face Mesh', annotated_image)

    # if cv2.waitKey(0) & 0xFF == 27:
        # cv2.destroyAllWindows()

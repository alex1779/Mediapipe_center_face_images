import cv2
# import math
import numpy as np
# import imutils
from datetime import datetime
from data.face_swap import get_landmark_points_face, indexes_triangles_face, sweep_faces, replacePartFace
from data.drawing import detect_side_face_minimal


def fix_side_face(image, landmark_list):
    global side
    global height
    global width
    side = detect_side_face_minimal(image=image, face_landmarks=landmark_list)
    height, width, _ = image.shape
    middle = int(width / 2)
    if side == 'left':
        # print('Detected left side')
        image1 = image[: height, : middle]
        image_flip = cv2.flip(image1, 1)
        height2, width2, _ = image_flip.shape
        image[:height, :middle] = image1
        image[:height2, width2:width] = image_flip
    else:
        # print('Detected right side')
        image1 = image[: height, middle:]
        image_flip = cv2.flip(image1, 1)
        height2, width2, _ = image_flip.shape
        image[:height, middle:] = image1
        image[:height2, :width2] = image_flip

    return image


def frontalFace(image, landmark_list):
    landmark_points = get_landmark_points_face(image, landmark_list)
    indexes_triangles = indexes_triangles_face(image, landmark_points)
    with np.load('data/landmarkpoints.npz') as data:
        img = data['arr_0']
        landmark_points_2 = data['arr_1']
    cannon_face = sweep_faces(image, landmark_points,
                              indexes_triangles, img, landmark_points_2)
    cannon_face = fix_side_face(cannon_face, landmark_list)
    return cannon_face


def fix_eye(img):
    left_keypoint_indices = [133, 173, 157, 158, 159, 160, 161,
                             246, 33, 7, 163, 144, 145, 153, 154, 155]
    right_keypoint_indices = [263, 466, 388, 387, 386, 385, 384,
                              398, 362, 382, 381, 380, 374, 373, 390, 249]
    if side == 'left':
        # print('Detected left side')
        img = replacePartFace(
            img, img, left_keypoint_indices, right_keypoint_indices)
    else:
        # print('Detected right side')
        img = replacePartFace(
            img, img, right_keypoint_indices, left_keypoint_indices)
    return img


def save_image(img):
    timestr = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
    cv2.imwrite('output/'+timestr+'.jpg', img)










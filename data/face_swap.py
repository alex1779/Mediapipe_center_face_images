# -*- coding: utf-8 -*-
"""
Created on Thu May 12 22:47:08 2022

@author: Ale
"""

import cv2
import numpy as np


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def get_landmark_points_face(image, face_landmarks):
    landmark_points_face = []
    height, width, _ = image.shape
    for index in range(468):
        x = int(face_landmarks.landmark[index].x * width)
        y = int(face_landmarks.landmark[index].y * height)
        landmark_points_face.append((x, y))
    return landmark_points_face


def indexes_triangles_face(image, landmark_points_face):
    indexes_triangles = []
    height, width, _ = image.shape
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    points = np.array(landmark_points_face, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points_face)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return indexes_triangles


def sweep_faces(img, landmark_points_face, indexes_triangles, img2, landmark_points_2):
    try:

        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(img2)
        points2 = np.array(landmark_points_2, np.int32)
        convexhull2 = cv2.convexHull(points2)

        # Triangulation of both faces
        for triangle_index in indexes_triangles:

            # Triangulation of the first face
            tr1_pt1 = landmark_points_face[triangle_index[0]]
            tr1_pt2 = landmark_points_face[triangle_index[1]]
            tr1_pt3 = landmark_points_face[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = img[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)
            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                               [tr1_pt2[0] - x, tr1_pt2[1] - y],
                               [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

            # Triangulation of second face
            tr2_pt1 = landmark_points_2[triangle_index[0]]
            tr2_pt2 = landmark_points_2[triangle_index[1]]
            tr2_pt3 = landmark_points_2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2
            cropped_tr2_mask = np.zeros((h, w), np.uint8)
            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # Reconstructing destination face
            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(
                img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(
                img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(
                img2_new_face_rect_area, warped_triangle)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)
        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
        image = cv2.seamlessClone(
            result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

        return image

    except:
        print('Cannot sweep face for this image.')
        return 0



def replacePartFace(imgSource, imgDest, ptsSource, ptsDest):

    with np.load('data/landmarkpoints.npz') as data:
        
        landmark_points = data['arr_1']
        indexes_triangles = data['arr_2']

    landmark_points_selected = []
    for idx, index in enumerate(ptsSource):
        PointA = int(landmark_points[index][0]), int(
            landmark_points[index][1])
        landmark_points_selected.append(PointA)

    landmark_points_selected2 = []
    for idx, index in enumerate(ptsDest):
        PointA = int(landmark_points[index][0]), int(
            landmark_points[index][1])
        landmark_points_selected2.append(PointA)

    img_gray = cv2.cvtColor(imgSource, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(imgDest, cv2.COLOR_BGR2GRAY)
    img2_new_face = np.zeros_like(imgDest)
    points = np.array(landmark_points_selected, np.int32)
    convexhull = cv2.convexHull(points)
    mask = np.zeros_like(img_gray)
    cv2.fillConvexPoly(mask, convexhull, 255)
    points2 = np.array(landmark_points_selected2, np.int32)
    convexhull2 = cv2.convexHull(points2)

    # # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points_selected)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    # Triangulation of both faces
    for triangle_index in indexes_triangles:

        # Triangulation of the first face
        tr1_pt1 = landmark_points_selected[triangle_index[0]]
        tr1_pt2 = landmark_points_selected[triangle_index[1]]
        tr1_pt3 = landmark_points_selected[triangle_index[2]]

        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = imgSource[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
        # Triangulation of second face
        tr2_pt1 = landmark_points_selected2[triangle_index[0]]
        tr2_pt2 = landmark_points_selected2[triangle_index[1]]
        tr2_pt3 = landmark_points_selected2[triangle_index[2]]

        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(
            warped_triangle, warped_triangle, mask=cropped_tr2_mask)
        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(
            img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(
            img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(
            warped_triangle, warped_triangle, mask=mask_triangles_designed)
        img2_new_face_rect_area = cv2.add(
            img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    img1_face_mask = np.zeros_like(img_gray)
    img1_head_mask = cv2.fillConvexPoly(img1_face_mask, convexhull, 255)
    img1_face_mask = cv2.bitwise_not(img1_head_mask)
    img2_head_noface = cv2.bitwise_and(imgDest, imgDest, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)
    
    return result
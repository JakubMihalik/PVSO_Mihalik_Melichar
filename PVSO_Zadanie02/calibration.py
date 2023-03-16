import glob
import cv2 as cv
import numpy as np

def calibrate(ximea: bool):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ROWS = 5
    COLS = 7

    # Arrays to store object points and image points from all the images.
    object_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((COLS * ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)  # 2D -> 1D (5,7,3) -> (35, 3)

    if ximea:
        images = glob.glob('Ximea_Images/*.jpg')  # read all images
    else:
        images = glob.glob('Webcam_Images/*.jpg')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret_val, corners = cv.findChessboardCorners(gray, (COLS, ROWS), None)
        # If found, add object points, image points (after refining them)
        if ret_val:
            object_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)
            # Draw and display the corners
            # cv.drawChessboardCorners(img, (COLS, ROWS), corners2, ret_val)
            # cv.imshow('img', img)
            # cv.waitKey(500)
    ret_val, camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors = cv.calibrateCamera(object_points, img_points, gray.shape[::-1], None, None)
    calculate_error(camera_matrix, object_points, img_points, distortion_coeffs, rotation_vectors, translation_vectors)
    return camera_matrix, distortion_coeffs


def calculate_error(camera_matrix, object_points, img_points, distortion_coeffs, rotation_vectors, translation_vectors):
    mean_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv.projectPoints(object_points[i], rotation_vectors[i], translation_vectors[i], camera_matrix,
                                         distortion_coeffs)
        error = cv.norm(img_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(object_points)))

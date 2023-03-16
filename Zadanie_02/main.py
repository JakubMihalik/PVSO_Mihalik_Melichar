import cv2 as cv
import numpy as np
import glob

WIDTH: int = 400
HEIGHT: int = 400


def process_image(width, height):
    ret, img = cam.read()
    image = cv.resize(img, (width, height))
    return image


def setup_camera():
    cam = cv.VideoCapture(0)
    return cam


cam = setup_camera()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ROWS = 5
COLS = 7

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((COLS * ROWS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) # 2D -> 1D (5,7,3) -> (35, 3)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('*.jpg') # read all images

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    retVal, corners = cv.findChessboardCorners(gray, (COLS, ROWS), None)
    # If found, add object points, image points (after refining them)
    if retVal:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (COLS, ROWS), corners2, retVal)
        # cv.imshow('img', img)
        # cv.waitKey(500)

# camera_matrix = [fx 0 cx; 0 fy cy; 0 0 1]
retVal, camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("fx = {0}\ncx = {1}\nfy = {2}\ncy = {3}".format(camera_matrix[0][0], camera_matrix[0][2], camera_matrix[1][1], camera_matrix[1][2]))
img = cv.imread('Ximea_Images/chessboard0.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, camera_matrix, distortion_coeffs, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rotation_vectors[i], translation_vectors[i], camera_matrix, distortion_coeffs)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )


while True:
    img = process_image(WIDTH, HEIGHT)
    dst = cv.undistort(img, camera_matrix, distortion_coeffs, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imshow('calibresult.png', dst)
    pressed = cv.waitKey(1)
    if pressed == ord('q'):
        exit(0)

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv.destroyAllWindows()

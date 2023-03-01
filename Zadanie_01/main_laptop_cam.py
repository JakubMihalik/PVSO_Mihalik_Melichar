import cv2 as cv
import numpy as np
import image_processing as ip

WIDTH: int = 400
HEIGHT: int = 400

def setup_camera():
    cam = cv.VideoCapture(0)
    return cam


def process_image(width, height):
    ret, img = cam.read()
    image = cv.resize(img, (width, height))
    return image

cam = setup_camera()


cur_index = 0
image_data = []

image_channels = 3
cols = 2
rows = 2

while cur_index < 4:
    processed_image = ip.process_image(cam, WIDTH, HEIGHT)
    pressed = cv.waitKey(1)
    if pressed == ord(' '):
        image_data.append(processed_image)
        cv.imwrite("Images/pekne_fotky_{0}.jpg".format(cur_index), processed_image)
        cur_index += 1
    elif pressed == ord('q'):
        exit(0)
    cv.imshow("Images/Image", processed_image)

# now concat the data but reshape them first to be 2x2 grid
result = ip.concat_images(np.array(image_data).reshape(cols, rows, HEIGHT, WIDTH, image_channels))

# save the image as mosaic.jpg
cv.imwrite("Images/mosaic.jpg", np.array(result))

# kernel
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

adj = ip.apply_kernel(cv.imread("Images/mosaic.jpg"), 1, kernel, WIDTH, HEIGHT)
adj = ip.apply_color(adj, 2, "red", WIDTH, HEIGHT)
adj = ip.rotate_image(adj, 3, WIDTH, HEIGHT)
cv.imwrite("Images/mosaic_with_kernel.jpg", adj)

# Post process mosaic
mosaic = cv.imread("Images/mosaic.jpg")

kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# kernel = np.array([[-4, -2, -4],
#                    [-2, 25, -2],
#                    [-4, -2, -4]])

# Print information about image
print("Dimensions(H x W): ", mosaic.shape[0], " x ", mosaic.shape[1])
print("Type: ", mosaic.dtype)
print("Size: ", mosaic.size)

# cv.imshow("image", mosaic)
# cv.waitKey()

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv.destroyAllWindows()

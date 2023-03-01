import cv2 as cv
import numpy as np


def process_image(cam, width, height):
    ret, img = cam.read()
    image = cv.resize(img, (width, height))
    return image


def concat_images(images):
    return cv.vconcat([cv.hconcat(img) for img in images])


def apply_kernel(images, image_number, kernel, width, height):
    assert image_number != 0
    # Calculate the row and column indices for the selected sub-image
    row_index = (image_number - 1) // 2
    col_index = (image_number - 1) % 2

    # Slice the 3D numpy array to extract the corresponding 2D numpy array
    img = images[row_index * height:(row_index + 1) * height, col_index * width:(col_index + 1) * width]
    filtered_image = cv.filter2D(img, -1, kernel)

    # Replace the original image with the filtered image in the 3D numpy array
    images[row_index * height:(row_index + 1) * height, col_index * width:(col_index + 1) * width] = filtered_image
    return images


def apply_color(images, image_number, color, width, height):
    color_switcher = {
        "blue": 0,
        "green": 1,
        "red": 2,
    }

    assert image_number != 0
    # Calculate the row and column indices for the selected sub-image
    row_index = (image_number - 1) // 2
    col_index = (image_number - 1) % 2

    # Slice the 3D numpy array to extract the corresponding 2D numpy array
    img = images[row_index * height:(row_index + 1) * height, col_index * width:(col_index + 1) * width]
    chosen_color = color_switcher.get(color, -1)

    for i in range(3):
        if i != chosen_color:
            img[:, :, i] = 0

    # Replace the original image with the filtered image in the 3D numpy array
    images[row_index * height:(row_index + 1) * height, col_index * width:(col_index + 1) * width] = img
    return images


def extract_image(images, image_number, height, width):
    row_index = (image_number - 1) // 2
    col_index = (image_number - 1) % 2
    extracted_image = images[row_index * height:(row_index + 1) * height, col_index * width:(col_index + 1) * width]
    return extracted_image


def rotate_image(images, image_number, height, width):
    row_index = (image_number - 1) // 2
    col_index = (image_number - 1) % 2

    extracted_image = images[row_index * height:(row_index + 1) * height, col_index * width:(col_index + 1) * width]

    # Create a new 2D list with the dimensions swapped
    rotated_tile = np.empty((height, width, 3))

    # Loop through the original image and copy each pixel to the new image
    for i in range(width):
        for j in range(height):
            # rotated_tile[j][i] = extracted_image[i][height-j-1]
            rotated_tile[i][j] = extracted_image[j][height-i-1]

    images[row_index * height:(row_index + 1) * height, col_index * width:(col_index + 1) * width] = np.array(rotated_tile)
    return images

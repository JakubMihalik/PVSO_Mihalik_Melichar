from ximea import xiapi

def setup_camera():
    cam = xiapi.Camera()
    cam.open_device()
    cam.set_exposure(1e5)
    cam.set_param("imgdataformat", "XI_RGB32")
    cam.set_param("auto_wb", 1)
    cam.start_acquisition()
    img = xiapi.Image()
    return cam, img

def process_image(width, height):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv.resize(image, (width, height))
    return image
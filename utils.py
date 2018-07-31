import glob
import os
import pickle
from os import path
import matplotlib.image as mpimg
import cv2
import numpy as np
from matplotlib import pyplot as plt

from consts import yellow_hsv_min, yellow_hsv_max


def calibrate(images_dir='camera_cal'):
    """
    Calibrate the camera given a directory containing calibration chessboards.
    :param images_dir: directory containing calibrated images.
    :param mode: video or image
    :return: calibration parameters
    """
    cache = '{}/data.pickle'.format(images_dir)

    if path.exists(cache):
        print('Loading cached calibration...')
        with open(cache, 'rb') as dump_file:
            calibration = pickle.load(dump_file)
    else:
        print('Computing camera calibration...')
        image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
        obj_points = []
        img_points = []
        nx, ny = 9, 6
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        for image_file in image_files:
            image = mpimg.imread(image_file)
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                # only append the points if ret is not 0
                obj_points.append(objp)
                img_points.append(corners)
        ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        calibration = (mtx, dist)
        with open(cache, 'wb') as dump_file:
            pickle.dump(calibration, dump_file)

    return calibration[0], calibration[1]


def undistort(image, mtx, dist, visualise=False):
    """
     Undistort an image given camera matrix and distortion coefficients.
    :param visualise:
    :param image: input image
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :return: undistorted image
    """
    undist = cv2.undistort(image, mtx, dist, newCameraMatrix=mtx)
    if visualise:
        f, axes = plt.subplots(1, 2)
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].set_axis_off()

        axes[1].imshow(undist, cmap='gray')
        axes[1].set_title('Undistorted Image')
        axes[1].set_axis_off()
        plt.show()
        f.savefig("output_images/undistort.png", format='png', bbox_inches='tight', transparent=True)
    return undist


def hsv_threshold(frame, min_values, max_values):
    """
    Threshold a color frame in HSV space
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, min_values, max_values)
    return mask.astype(bool)


def get_binary_from_equalized_grayscale(frame):
    """
    Apply histogram equalization to an input frame, threshold it and return the (binary) result.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    eq_global = cv2.equalizeHist(gray)
    _, mask = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
    return mask


def mag_threshold(image, sobel_kernel=9):
    """
    Apply edge detection with sobel kernel and apply threshold filter by magnitude
    :param image:
    :param sobel_kernel:
    :return:
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    _, sobel_mag = cv2.threshold(gradmag, thresh=50, maxval=1, type=cv2.THRESH_BINARY)
    return sobel_mag.astype(bool)


def binarize(img, visualise=False):
    h, w = img.shape[:2]

    # create all zeros image with shape of original image
    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    # highlight yellow lines by thresholding hsv
    hsv_yellow_mask = hsv_threshold(img, yellow_hsv_min, yellow_hsv_max)

    # highlight white lines by thresholding the equalized frame
    eq_white_mask = get_binary_from_equalized_grayscale(img)

    # thresholded gradients to get Sobel binary mask
    sobel_mask = mag_threshold(img)

    # combine the three steps results by logical or
    binary = np.logical_or(binary, hsv_yellow_mask)
    binary = np.logical_or(binary, eq_white_mask)
    out = np.logical_or(binary, sobel_mask).astype(np.uint8)
    return out.astype(np.uint8)


def birdeye(img, visualise=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param visualise: if True, show the transformation result
    :return: warped image, and both forward and backward transformation matrices
    """
    h, w = img.shape[:2]  # 720x1280

    src = np.float32([[w, h - 10],
                      [0, h - 10],
                      [540, 460],
                      [750, 460]])
    dst = np.float32([[w, h],
                      [0, h],
                      [0, 0],
                      [w, 0]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_matrix = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR)

    if visualise:
        f, axes = plt.subplots(1, 2)
        f.set_facecolor('white')
        axes[0].set_title('Before perspective transform')
        axes[0].imshow(img, cmap='gray')
        for point in src:
            axes[0].plot(*point, '.')

        axes[1].set_title('After perspective transform')
        axes[1].imshow(warped, cmap='gray')
        for point in dst:
            axes[1].plot(*point, '.')

        for axis in axes:
            axis.set_axis_off()
        plt.show()
        f.savefig('output_images/birdeye.png', format='png', bbox_inches='tight', transparent=True)

    return warped, matrix, inverse_matrix


if __name__ == '__main__':

    mtx, dist = calibrate('camera_cal')

    # show result on test images
    for test_img in glob.glob('camera_cal/*.jpg'):
        img = mpimg.imread(test_img)
        img_undistorted = undistort(img, mtx, dist, True)
        img_binary = binarize(img_undistorted, True)
        img_birdeye, matrix, inverse_matrix = birdeye(img_undistorted, True)


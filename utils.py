import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
# % matplotlib inline

yellow_HSV_th_min = np.array([0, 70, 70])
yellow_HSV_th_max = np.array([50, 255, 255])

def calibrate(images_dir='camera_cal'):
    """
    Calibrate the camera given a directory containing calibration chessboards.
    :param images_dir: directory containing calibrated images.
    :return: calibration parameters
    """
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    obj_points = []
    img_points = []
    nx, ny = 9, 6
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for image_file in image_files:
        image = cv2.imread(image_file)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return mtx, dist


def undistort(image, mtx, dist):
    """
     Undistort an image given camera matrix and distortion coefficients.
    :param image: input image
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :return: undistorted image
    """
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist


def HSV_threshold(frame, min_values, max_values):
    """
    Threshold a color frame in HSV space
    """
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    min_th_ok = np.all(HSV > min_values, axis=2)
    max_th_ok = np.all(HSV < max_values, axis=2)

    out = np.logical_and(min_th_ok, max_th_ok)

    return out


def mag_threshold(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
#     binary_output = np.zeros_like(gradmag)
#     binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    _, sobel_mag = cv2.threshold(gradmag, 50, 1, cv2.THRESH_BINARY)

    return sobel_mag.astype(bool)



def binarize(img):
    h, w = img.shape[:2]
    binary = np.zeros(shape=(h, w), dtype=np.uint8)
    out = HSV_threshold(img, yellow_HSV_th_min, yellow_HSV_th_max)
    binary = np.logical_or(binary, out)
    eq_white_mask = get_binary_from_equalized_grayscale(img)
    print(eq_white_mask.shape)
    binary = np.logical_or(binary, eq_white_mask)
    out = mag_threshold(img)
    out = np.logical_or(binary, out)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closing

def get_binary_from_equalized_grayscale(frame):
    """
    Apply histogram equalization to an input frame, threshold it and return the (binary) result.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eq_global = cv2.equalizeHist(gray)

    _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

    return th


def birdeye(img, verbose=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param verbose: if True, show the transformation result
    :return: warped image, and both forward and backward transformation matrices
    """
    h, w = img.shape[:2]

    src = np.float32([[w, h-10],    # br
                      [0, h-10],    # bl
                      [546, 460],   # tl
                      [732, 460]])  # tr
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    print("birdeye:",img.shape)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)


    if verbose:
        f, axarray = plt.subplots(1, 2)
        f.set_facecolor('white')
        axarray[0].set_title('Before perspective transform')
        axarray[0].imshow(img, cmap='gray')
        for point in src:
            axarray[0].plot(*point, '.')
        axarray[1].set_title('After perspective transform')
        axarray[1].imshow(warped, cmap='gray')
        for point in dst:
            axarray[1].plot(*point, '.')
        for axis in axarray:
            axis.set_axis_off()
        plt.show()

    return warped, M, Minv


if __name__ == '__main__':

    mtx, dist = calibrate('camera_cal')

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)

        img_undistorted = undistort(img, mtx, dist)

        img_binary = binarize(img_undistorted)
        # plt.imshow(img_binary, cmap="gray")

        img_birdeye, M, Minv = birdeye(img_binary, verbose=True)

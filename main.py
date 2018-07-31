import cv2
import glob
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
from utils import calibrate, undistort, binarize, birdeye, plt

from line_utils import Line, offset_from_lane_center, find_lane_by_sliding_windows, find_lane_by_previous_fits, \
    draw_back_onto_the_road


def process_pipeline(frame, keep_state=True):
    """
    Apply whole lane detection pipeline to an input color frame.
    :param frame: input color frame
    :param keep_state: if True, lane-line state is conserved (this permits to average results)
    :return: output blend with detected lane overlaid
    """

    global line_lt, line_rt, processed_frames

    # undistort the image using coefficients found in calibration
    undistorted_img = undistort(frame, mtx, dist)

    # binarize the frame and highlight lane lines
    binarized_img = binarize(undistorted_img)

    # perspective transform to obtain bird's eye view
    birdeye_img, matrix, inversed_matrix = birdeye(binarized_img, visualise=False)

    #  2 order polynomial curve fit onto lane lines found
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        find_lane_by_previous_fits(birdeye_img, line_lt, line_rt, visualise=False)
    else:
        find_lane_by_sliding_windows(birdeye_img, line_lt, line_rt, n_windows=9, visualise=False)

    # compute offset in meter from center of the lane
    offset_meter = offset_from_lane_center(line_lt, line_rt, frame_width=frame.shape[1])

    # draw the surface enclosed by lane lines back onto the original frame
    blend_on_road = draw_back_onto_the_road(undistorted_img, inversed_matrix, line_lt, line_rt, keep_state)
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (60, 60), font, 1,
                (255, 255, 255), 2)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (60, 90), font, 1,
                (255, 255, 255), 2)

    processed_frames += 1

    return blend_on_road


if __name__ == '__main__':

    # step 1: calibrate the camera
    mtx, dist = calibrate('camera_cal')
    mode = "video"
    processed_frames = 0
    line_lt, line_rt = Line(buffer_len=10), Line(buffer_len=10)

    if mode == "image":
        for test_img in glob.glob('test_images/*.jpg'):
            img = mpimg.imread(test_img)
            img_out = process_pipeline(img, True)
            plt.imshow(img_out)
            plt.show()

    elif mode == "video":
        selector = 'project'
        clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(process_pipeline)
        clip.write_videofile('out_{}.mp4'.format(selector), audio=False)

## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/birdeye.png "Warp Example"
[image5]: ./output_images/lane_detect.png "Fit Visual"
[image6]: ./output_images/blend_on_road.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.   

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #13 through #50 of the file called `utils.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

To avoid calibration every time, I dump the calibration result to a pickle file: data.pickle, so as to load it directly next time if the pickle file exsists. 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HSV color (for yellow lane line) and gradient magnitude (for edge detection) thresholds to generate a binary image (thresholding steps at lines #125 through #136 in `utils.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `birdeye()`, which appears in lines #179 through #188 in the file `utils.py` (utils.py).  The `birdeye()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
h, w = img.shape[:2]  # 720x1280
src = np.float32([[w, h - 10],
                    [0, h - 10],
                    [540, 460],
                    [750, 460]])
dst = np.float32([[w, h],
                    [0, h],
                    [0, 0],
                    [w, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1280, 710      | 1280, 720        | 
| 0, 710        | 0, 720
| 540, 460      | 0, 0      |
| 750, 460     | 1280, 0      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

If we have a brand new frame, and we never identified where the lane-lines are, we can use the two highest peaks from our histogram as a starting point for determining where the lane lines are, and then use sliding windows moving upward in the image (further along the road) to determine where the lane lines go. This is implemented in line_utils.find_lane_by_sliding_windows().

In the next frame of video, we don't need to do a blind search again, but instead we can just search in a margin around the previous line position, like in the above image. So, once we know where the lines are in one frame of video, you can do a highly targeted search for them in the next frame. This is implemented in line_utils.find_lane_by_previous_fits().

In order to keep track of detected lines across successful frames, I employ a class defined in line_utils.Line

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Offset from center of the lane is computed in offset_from_lane_center() defined in line_utils.py. 
Suppose the postion of car is fixed, the offset can be approximately calculated as the distance between the center of the image and the midpoint at the bottom of the two lane-lines detected on the image.

In the previous lane-line detection phase, a 2nd order polynomial is fitted to each lane-line using np.polyfit(), which returns the 3 coefficients that describe the curve. From this coefficients, following this equation, we can compute the radius of curvature of the curve. This is implemented as property of the Line class in line_utils.py.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `line_utils.py` in the function `draw_back_onto_the_road()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./out_project.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline might fail if the binarisation for the image couldn't get correct lane pixels. So a combination of theresholds for yellow/white color and edge detection is crucial to the result. Also, when the lane disappears in the image, using pixels found in previous frames is important. I took quite a while for understanding that part in the course.

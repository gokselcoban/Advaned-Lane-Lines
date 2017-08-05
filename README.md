**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Change color map to HLS, create a thresholded binary image with using gradient of L-channel and S-channel.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./report/calibration.jpg "Undistortion"
[image2]: ./report/undistort.jpg "Road Transformed"
[image3]: ./report/preprocessing.jpg "Binary Example"
[image4]: ./report/steps.jpg "Warp Example"
[image5]: ./report/result.jpg "Output"
[image6]: ./report/sliding_window.jpg "Fit Visual"
[image7]: ./report/after_sliding.jpg "Recent Fitted"
[image8]: ./report/debug.jpg "Output"

[video1]: ./report/project_video_output.mp4 "Video"

### Camera Calibration

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

The code for this step is contained in the `calibration.py`.  

![alt text][image1]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

![alt text][image2]

#### 2. Prepoccesing

I used a combination of color and gradient thresholds to generate a binary image.

The code for this step is contained in the `calibration.py`.
![alt text][image3]

#### 3. Perspective transform and finding the firt line.

I chose the hardcode the source and destination points in the following manner:

```python
  src_left_buttom = (np.int(xsize*0.145), np.int(ysize))
  src_left_top = (np.int(xsize*0.4), np.int(ysize*0.70))
  src_rigth_top = (np.int(xsize*0.6), np.int(ysize*0.70))
  src_right_buttom = (np.int(xsize*0.88), np.int(ysize))

  dst_left_buttom = (np.int(xsize * 0.20), np.int(ysize))
  dst_left_top = (np.int(xsize * 0.20), np.int(0))
  dst_rigth_top = (np.int(xsize * 0.80), np.int(0))
  dst_right_buttom = (np.int(xsize * 0.80), np.int(ysize))
```

The code for this step is contained in the `line_finding.py`.


![alt text][image4]
![alt text][image5]

#### 4. Sliding windows and fitting a polynomial you identified lane-line pixels and fit their positions with a polynomial?

Firstly, I take a histogram for lower half of the binary warped image. Secondly, I select peak points of the histogram to determine starting points of the sliding windows. After that, new positions of sliding windows are selected iteratively according to density of light colors. Finally, detected pixels inside the box are fitted to second order polynomial.

![alt text][image6]

If I detected the lane lines for previous pixels, I searching the pixels around of the average of last fitted lines.
![alt text][image7]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

I have tried to fit lane lines to second order polynomial. These polynomials are also similar to arc of a circle. So when radius of the curvature is small, it means that the curve is sharp.

I calculated the car position assuming the camera is in the middle of the car and center of the lane is middle point of two lines at 700th px.

#### 6. Result

![alt text][image8]

---

### Pipeline (video)

Here's a [link to my video result](./report/project_video_output.mp4)

---

### Discussion

My implementation gives satisfying result in good light conditions but it could be fail on shadow. To overcome this problem, another preprocessing technique  can be used or knowledge from previous frames can be used better.

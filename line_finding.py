from preprocessing import pipeline
import pickle
import cv2
import numpy as np
from line import Line
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Create lines
left_line = Line()
right_line = Line()


def process_image(img):
    img_size = (img.shape[1], img.shape[0])
    xsize = img.shape[1]
    ysize = img.shape[0]

    # Undistortion
    undistort = cv2.undistort(img, mtx, dist, None, mtx)

    # Prepoccesing pipeline
    binary = pipeline(img)

    # Center of the image according to X dimension
    midpoint = np.int(xsize / 2)

    # Determine source and destionation points for perspective transform
    src_left_buttom = (np.int(xsize*0.145), np.int(ysize))
    src_left_top = (np.int(xsize*0.4), np.int(ysize*0.70))
    src_rigth_top = (np.int(xsize*0.6), np.int(ysize*0.70))
    src_right_buttom = (np.int(xsize*0.88), np.int(ysize))

    dst_left_buttom = (np.int(xsize * 0.20), np.int(ysize))
    dst_left_top = (np.int(xsize * 0.20), np.int(0))
    dst_rigth_top = (np.int(xsize * 0.80), np.int(0))
    dst_right_buttom = (np.int(xsize * 0.80), np.int(ysize))

    src = np.float32([[src_left_buttom[0], src_left_buttom[1]],
                      [src_left_top[0], src_left_top[1]],
                      [src_rigth_top[0], src_rigth_top[1]],
                      [src_right_buttom[0], src_right_buttom[1]]])

    dst = np.float32([[dst_left_buttom[0], dst_left_buttom[1]],
                      [dst_left_top[0], dst_left_top[1]],
                      [dst_rigth_top[0], dst_rigth_top[1]],
                      [dst_right_buttom[0], dst_right_buttom[1]]])

    # Calculate transformation matrices
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Wrap processed binary image
    binary_warped = cv2.warpPerspective(
        binary, M, img_size, flags=cv2.INTER_LINEAR)


    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create an output image to draw on and  visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    if not (left_line.detected & right_line.detected):
    # if not (False):
        # Take a histogram of the bottom half of the image
        # histogram = np.sum(
        #     binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        histogram = np.sum(
            binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 12
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin

            # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
            #               (win_xleft_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

                # Stop when the line goes to out of image
                if leftx_current - margin < 0:
                    break

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xright_low, win_y_low),
            #               (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

                # Stop when the line goes to out of image
                if rightx_current + margin > xsize:
                    break

        # Concatenate the arrays of indices
        right_lane_inds = np.concatenate(right_lane_inds)
        left_lane_inds = np.concatenate(left_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        try:
            # Fit a second order polynomial
            left_fit = np.polyfit(lefty, leftx, 2)
            left_line.current_fit.append(left_fit)
            left_line.detected = True

        except TypeError:
            left_fit = left_line.get_best_fit()
            left_line.detected = False

        except ValueError:
            left_fit = left_line.get_best_fit()
            left_line.detected = False

        try:
            # Fit a second order polynomial
            right_fit = np.polyfit(righty, rightx, 2)
            right_line.current_fit.append(right_fit)
            right_line.detected = True

        except TypeError:
            right_fit = right_line.get_best_fit()
            right_line.detected = False

        except ValueError:
            right_fit = right_line.get_best_fit()
            right_line.detected = False

    else:
        # Assume you now have a new warped binary image
        # you approximately know where the lane from past frames
        margin = 100

        left_fit = left_line.get_best_fit()
        right_fit = right_line.get_best_fit()

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # # Create an image to draw on and an image to show the selection window
        # # out_img = np.dstack(
        # #     (binary_warped, binary_warped, binary_warped)) * 255
        # window_img = np.zeros_like(out_img)
        # # Color in left and right line pixels
        # out_img[nonzeroy[left_lane_inds],
        #         nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds],
        #         nonzerox[right_lane_inds]] = [0, 0, 255]
        #
        # ploty = np.linspace(0, 719, num=720)
        # left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        # right_fitx = right_fit[0] * ploty**2 + \
        #     right_fit[1] * ploty + right_fit[2]
        #
        # # Generate a polygon to illustrate the search window area
        # # And recast the x and y points into usable format for cv2.fillPoly()
        # left_line_window1 = np.array(
        #     [np.transpose(np.vstack([left_fitx - margin, ploty]))])
        # left_line_window2 = np.array(
        #     [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        # left_line_pts = np.hstack((left_line_window1, left_line_window2))
        # right_line_window1 = np.array(
        #     [np.transpose(np.vstack([right_fitx - margin, ploty]))])
        # right_line_window2 = np.array(
        #     [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        # right_line_pts = np.hstack((right_line_window1, right_line_window2))
        #
        # fig = plt.figure(figsize=(12.8, 7.2))
        # # Draw the lane onto the warped blank image
        # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        # plt.imshow(result)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # # plt.show()

        try:
            # Fit a second order polynomial
            left_fit = np.polyfit(lefty, leftx, 2)
            left_line.current_fit.append(left_fit)
            left_line.detected = True

        except TypeError:
            left_fit = left_line.get_best_fit()
            left_line.detected = False

        except ValueError:
            left_fit = left_line.get_best_fit()
            left_line.detected = False

        try:
            # Fit a second order polynomial
            right_fit = np.polyfit(righty, rightx, 2)
            right_line.current_fit.append(right_fit)
            right_line.detected = True

        except TypeError:
            right_fit = right_line.get_best_fit()
            right_line.detected = False

        except ValueError:
            right_fit = right_line.get_best_fit()
            right_line.detected = False

    ploty = np.linspace(0, 719, num=720)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Sample the distance between two lines
    diff = [right_fitx[100] - left_fitx[100],
            right_fitx[300] - left_fitx[300],
            right_fitx[500] - left_fitx[500],
            right_fitx[700] - left_fitx[700]]

    # The sampled distance should be smilar and the lines should not overlap
    if np.max(diff) - np.min(diff) > 150 or np.min(diff)<450 :
        left_line.detected = False
        right_line.detected = False
        left_fit = left_line.get_best_fit()
        right_fit = right_line.get_best_fit()
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]


    # Calculate the pixel curve radius
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # pixcel calculation
    # left_curverad = (
    #     (1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
    # right_curverad = (
    #     (1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
    #
    # print(left_curverad, right_curverad)
    # # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    try:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty* ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                               left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                                right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

        # Now our radius of curvature is in meters
        average_curverad = np.mean([left_curverad, right_curverad])
        curvature_text = str(average_curverad) + ' m'
        # print(curvature_text)

        # Example values: 632.1 m    626.2 m
        left_line.radius_of_curvature = left_curverad
        right_line.radius_of_curvature = right_curverad
    except TypeError:
        curvature_text = ""

    # Calculate car position
    lane_center = np.mean([left_fitx[700], right_fitx[700]])
    distance = lane_center - midpoint
    distance = distance * xm_per_pix

    car_position = str()
    if distance > 0:
        car_position = 'LEFT :' + str(distance) + ' m'
    else:
        car_position = 'RIGHT :' + str(distance) + ' m'

    # Fit a second order polynomial to pixel positions
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    left_line.allx = pts_left
    right_line.ally = pts_right

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)

    # Add the curvature of the lane and vehicle position
    cv2.putText(result, car_position, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(result, curvature_text, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return result


video = "challenge_video"
white_output = video + '_output.mp4'
clip1 = VideoFileClip(video + ".mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

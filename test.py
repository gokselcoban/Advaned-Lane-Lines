from preprocessing import pipeline
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

images = glob.glob('test_images/*.jpg')
for idx, fname in enumerate(images):
    # Test undistortion on an image
    img = mpimg.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    xsize = img.shape[1]
    ysize = img.shape[0]
    midpoint = np.int(xsize/2)
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    binary = pipeline(undistort)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Undistort', fontsize=30)

    # src_left_buttom = (np.int(xsize*0.145), np.int(ysize))
    # src_left_top = (np.int(xsize*0.46), np.int(ysize*0.63))
    # src_rigth_top = (np.int(xsize*0.54), np.int(ysize*0.63))
    # src_right_buttom = (np.int(xsize*0.88), np.int(ysize))

    src_left_buttom = (np.int(xsize*0.145), np.int(ysize))
    src_left_top = (np.int(xsize*0.4), np.int(ysize*0.70))
    src_rigth_top = (np.int(xsize*0.6), np.int(ysize*0.70))
    src_right_buttom = (np.int(xsize*0.88), np.int(ysize))

    dst_left_buttom = (np.int(xsize*0.20), np.int(ysize))
    dst_left_top = (np.int(xsize*0.20), np.int(0))
    dst_rigth_top = (np.int(xsize*0.80), np.int(0))
    dst_right_buttom = (np.int(xsize*0.80), np.int(ysize))

    src = np.float32([[src_left_buttom[0], src_left_buttom[1]],
                       [src_left_top[0], src_left_top[1]],
                       [src_rigth_top[0], src_rigth_top[1]],
                       [src_right_buttom[0], src_right_buttom[1]]])

    dst = np.float32([[dst_left_buttom[0], dst_left_buttom[1]],
                       [dst_left_top[0], dst_left_top[1]],
                       [dst_rigth_top[0], dst_rigth_top[1]],
                       [dst_right_buttom[0], dst_right_buttom[1]]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    binary_warped = cv2.warpPerspective(binary, M, img_size, flags=cv2.INTER_LINEAR)
    color_warped = cv2.warpPerspective(undistort, M, img_size, flags=cv2.INTER_LINEAR)


    cv2.line(undistort, src_left_buttom, src_left_top, (255, 0, 0), thickness=2, lineType=8)
    cv2.line(undistort, src_left_top, src_rigth_top, (255, 0, 0), thickness=2, lineType=8)
    cv2.line(undistort, src_rigth_top, src_right_buttom, (255, 0, 0), thickness=2, lineType=8)
    cv2.line(undistort, src_right_buttom, src_left_buttom, (255, 0, 0), thickness=2, lineType=8)


    ax2.imshow(undistort, cmap='gray')
    ax2.set_title('Source region', fontsize=30)

    ax3.imshow(color_warped, cmap='gray')
    ax3.set_title('Color warped', fontsize=30)

    ax4.imshow(binary, cmap='gray')
    ax4.set_title('Binary', fontsize=30)

    ax5.imshow(binary_warped, cmap='gray')
    ax5.set_title('Binary warped', fontsize=30)

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])+midpoint

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Choose the number of sliding windows
    nwindows = 12

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

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
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    diff = [left_fitx[300] -right_fitx[300],
            left_fitx[350] -right_fitx[350],
            left_fitx[600] -right_fitx[600]]
    print( np.max(diff) - np.min(diff))

    # print(left_fitx[200])
    # print(left_fitx[400])
    # print(left_fitx[600])
    # print(right_fitx[200])
    # print(right_fitx[400])
    # print(right_fitx[600])
    # print(left_fitx[200] -right_fitx[200])
    # print(left_fitx[400] -right_fitx[400])
    # print(left_fitx[600] -right_fitx[600])

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    ax6.imshow(out_img)
    ax6.set_title('Sliding windows', fontsize=30)
    plt.savefig('report/temp.jpg')

    # plt.imshow(binary_warped, cmap='gray')
    # plt.plot(left_fitx, ploty, color='red')
    # plt.plot(right_fitx, ploty, color='red')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    ploty = np.linspace(0, 719, num=720)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    # Fit a second order polynomial to pixel positions
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    plt.figure()
    plt.imshow(result)
    plt.savefig('report/result.jpg')
    break

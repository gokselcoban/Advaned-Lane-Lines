import cv2
import numpy as np
import matplotlib.pyplot as plt

def pipeline(img, s_thresh=(75, 255), sx_thresh=(20, 100)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    # Threshold color channel
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    combined = np.zeros_like(sxbinary)
    combined[(sxbinary == 1) | (s_binary == 1)] = 1

    # f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 10))
    # ax1.imshow(img)
    # ax1.set_title('Undistorted', fontsize=30)
    #
    # ax2.imshow(l_channel, cmap='gray')
    # ax2.set_title('L Channel', fontsize=30)
    #
    # ax3.imshow(s_channel, cmap='gray')
    # ax3.set_title('S Channel', fontsize=30)
    #
    # ax4.imshow(sxbinary, cmap='gray')
    # ax4.set_title('Sx binary', fontsize=30)
    #
    # ax5.imshow(s_binary, cmap='gray')
    # ax5.set_title('S binary', fontsize=30)
    #
    # ax6.imshow(combined, cmap='gray')
    # ax6.set_title('Result', fontsize=30)
    #
    # plt.savefig('report/preprocessing.jpg')

    return combined

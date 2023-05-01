import numpy as np
import cv2
import argparse
import sys
from calibration_store import load_stereo_coefficients

def crop_mid(image,roi):
    x1, y1, x2, y2 = roi
    dst = image[y1:y2, x1:x2]
    # dst = image[y:y + h, x:x + w]
    return dst

# bundle adjustment?
def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 10  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=12*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,

        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImgN = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImgN = np.uint8(filteredImg)
    confMap = wls_filter.getConfidenceMap()

    return filteredImgN,filteredImg, confMap


if __name__ == '__main__':
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument('--left_source', type=str, required=True, help='Left video or v4l2 device name')
    parser.add_argument('--right_source', type=str, required=True, help='Right video or v4l2 device name')
    parser.add_argument('--is_real_time', type=int, required=True, help='Is it camera stream or video')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save Images')
    args = parser.parse_args()

    base_dir = args.save_dir
    # is camera stream or video
    if args.is_real_time:
        cap_left = cv2.VideoCapture(args.left_source, cv2.CAP_V4L2)
        cap_right = cv2.VideoCapture(args.right_source, cv2.CAP_V4L2)
    else:
        cap_left = cv2.VideoCapture(args.left_source)
        cap_right = cv2.VideoCapture(args.right_source)

    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params
    # print(K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q)
    if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
        print("Can't opened the streams!")
        sys.exit(-9)

    # Change the resolution in need
    # cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
    # cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float
    #
    # cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
    # cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float
    kkk = 1253
    while True:  # Loop until 'q' pressed or stream ends
        # Grab&retreive for sync images

        if not (cap_left.grab() and cap_right.grab()):
            print("No more frames")
            break

        _, leftFrame = cap_left.retrieve()
        _, rightFrame = cap_right.retrieve()
        height, width, channel = leftFrame.shape  # We will use the shape for remap
        # scale = 1
        # height = height*scale
        # width = width*scale
        # roiL = np.array([1000,400,1200,500])*scale
        # roiR = np.array([1000, 400, 1200, 500]) * scale

        # Undistortion and Rectification part!
        # print(K1, D1, R1, P1, (width, height))
        # alpha = 1
        # K1, roiL = cv2.getOptimalNewCameraMatrix(K1, D1, (width, height), alpha, (width, height))
        # K2, roiR = cv2.getOptimalNewCameraMatrix(K2, D2, (width, height), alpha, (width, height))
        # undistort
        dstL = cv2.undistort(leftFrame, K1, D1)
        dstR = cv2.undistort(rightFrame, K2, D2)
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        # left_rectified = crop_mid(left_rectified, roiL)
        # right_rectified = crop_mid(right_rectified, roiR)
        # x, y, w, h = roiL
        # print(roiL)
        # left_rectified = left_rectified[y:y + h, x:x + w]
        # x, y, w, h = roiR
        # right_rectified = right_rectified[y:y + h, x:x + w]

        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        disparity_image, disparity_mat, confMap = depth_map(gray_left, gray_right)  # Get the disparity map

        # Show the images
        # keypoints = [(431,674),(336,587),(409,67),(940,707)]
        # xy
        print('************************* image_'+str(kkk)+" *************************")
        # for iii, kp in enumerate(keypoints):
        #     print("kp_"+str(iii)+"_"+str(kp)+':',disparity_mat[kp[1],kp[0]])
        # break
        # cv2.imshow('left(L)', leftFrame)
        # cv2.imshow('right(R)', rightFrame)
        # cv2.imshow('Disparity', disparity_image)

        catImg = cv2.hconcat([left_rectified,right_rectified])
        # print(left_rectified.shape)
        # print(right_rectified.shape)
        # print(catImg.shape)
        cv2.imwrite(base_dir + 'recPair\\recPair_%.2d.png'%kkk, catImg)
        cv2.imwrite(base_dir + 'disp\\confMap_%.2d.png'%kkk, confMap)
        cv2.imwrite(base_dir + 'rec\\recL_%.2d.png'%kkk, left_rectified)
        cv2.imwrite(base_dir + 'rec\\recR_%.2d.png'%kkk, right_rectified)
        cv2.imwrite(base_dir + 'undist\\undistL_%.2d.png'%kkk, dstL)
        cv2.imwrite(base_dir + 'undist\\undistR_%.2d.png'%kkk, dstR)
        cv2.imwrite(base_dir+'disp\\disparity_%.2d.png'%kkk,disparity_image)
        kkk +=1

        # cv2.waitKey(10000)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
            break
        # if kkk ==10:
        #     break

    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()



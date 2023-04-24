import os.path

import numpy as np
import cv2
import glob
import argparse
import sys
from calibration_store import load_coefficients, save_stereo_coefficients
import pickle

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = (2160, 3840)


# argphrase

cam_no = str(1)

# define world coord
width=7
height=7
width_mm = 223
length_mm = 330
carpet_3D = np.zeros((height,width,3))
for h in range(height):
    for w in range(width):
        carpet_3D[h,w,0] = w*width_mm
        carpet_3D[h,w,1] = h*length_mm
carpet_3D = carpet_3D.reshape([-1,3])





# find carpet corner
if False:
    image_name = 'camera1_calib.jpg'
    fname = os.path.join('data','2022_05_16',image_name)
    img = cv2.imread(fname)

    print(fname, end=',')
    # print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

    # If found, add object points, image points (after refining them)
    if ret:
        print('found')
        # print(corners)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        carpet_2D = corners2

        # Draw and display the corners
        # Show the image to see if pattern is found ! imshow function.
        img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
    else:
        print("no corner found")
        # print(gray)
        dim = (int(img.shape[1]/4),int(img.shape[0]/4))
        print(img.shape,dim)
        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow(fname,img)
        cv2.waitKey(800)
        cv2.destroyWindow(fname)
        # cv
else:
    if cam_no == '1':
        # camera new left / 1
        carpet_2D = np.array([
        [1660.192308,	1126.884615],
        [1572.961538,	1153.192308],
        [1473.269231,	1182.269231],
        [1370.807692,	1218.269231],
        [1264.192308,	1247.346154],
        [1153.423077,	1280.576923],
        [1037.115385,	1319.346154],
        [1775.115385,	1161.5],
        [1680.961538,	1189.192308],
        [1585.423077,	1223.807692],
        [1484.346154,	1259.807692],
        [1366.653846,	1297.192308],
        [1257.269231,	1335.961538],
        [1129.884615,	1376.115385],
        [1905.269231,	1203.038462],
        [1813.884615,	1236.269231],
        [1712.807692,	1272.269231],
        [1606.192308,	1309.653846],
        [1494.038462,	1355.346154],
        [1373.576923,	1395.5],
        [1240.653846,	1441.192308],
        [2046.5	   , 1247.346154],
        [1953.730769,	1284.730769],
        [1854.038462,	1323.5],
        [1747.423077,	1367.807692],
        [1635.269231,	1412.115385],
        [1505.115385,	1463.346154],
        [1372.192308,	1517.346154],
        [2207.115385,	1293.038462],
        [2118.5	   , 1340.115385],
        [2013.269231,	1380.269231],
        [1905.269231,	1431.5],
        [1793.115385,	1486.884615],
        [1662.961538,	1540.884615],
        [1525.884615,	1605.961538],
        [2382.961538,	1348.423077],
        [2297.115385,	1395.5],
        [2201.576923,	1448.115385],
        [2092.192308,	1503.5],
        [1975.884615,	1565.807692],
        [1849.884615,	1632.269231],
        [1700.346154,	1702.884615],
        [2575.423077,	1406.576923],
        [2496.5	   , 1459.192308],
        [2400.961538,	1517.346154],[2299.884615,	1586.576923],[2187.730769,	1654.423077],[2058.961538,	1734.730769],[1913.576923,	1817.807692]
        ])
    elif cam_no == '8':
        carpet_2D =np.array([
            [2660.75,	1321.25],
            [2557.25,	1274.75],
            [2461.25,	1232.75],
            [2363.75,	1192.25],
            [2276.75,	1153.25],
            [2197.25,	1112.75],
            [2116.25,	1079.75],
            [2539.25,	1399.25],
            [2434.25,	1349.75],
            [2332.25,	1300.25],
            [2237.75,	1250.75],
            [2146.25,	1211.75],
            [2062.25,	1169.75],
            [1988.75,	1130.75],
            [2402.75,	1487.75],
            [2291.75,	1424.75],
            [2192.75,	1369.25],
            [2093.75,	1321.25],
            [2008.25,	1273.25],
            [1925.75,	1228.25],
            [1849.25,	1187.75],
            [2242.25,	1579.25],
            [2134.25,	1511.75],
            [2035.25,	1450.25],
            [1942.25,	1393.25],
            [1855.25,	1342.25],
            [1775.75,	1291.25],
            [1702.25,	1246.25],
            [2071.25,	1684.25],
            [1961.75,	1607.75],
            [1862.75,	1537.25],
            [1772.75,	1472.75],
            [1687.25,	1417.25],
            [1612.25,	1360.25],
            [1541.75,	1310.75],
            [1871.75,	1796.75],
            [1768.25,	1712.75],
            [1672.25,	1633.25],
            [1586.75,	1561.25],
            [1510.25,	1495.25],
            [1435.25,	1435.25],
            [1373.75,	1379.75],
            [1651.25,	1915.25],
            [1550.75,	1820.75],
            [1465.25,	1735.25],
            [1387.25,	1654.25],
            [1312.25,	1582.25],
            [1247.75,	1514.75],
            [1189.25,	1450.25]
            ])
# load camera intrinsic
mono_cali_file = os.path.join('data','1_18','left_cam.yml')
# only left seems to be performing good, no need to change
# mono_cali_file = os.path.join('data','1_18','right_cam.yml')
camera_matrix, dist_coeffs = load_coefficients(mono_cali_file)

# solve pnp
# n = 45
# carpet_3D = carpet_3D[:n]
# carpet_2D = carpet_2D[:n]

success, rotation_vector, translation_vector = cv2.solvePnP(carpet_3D, carpet_2D, camera_matrix, dist_coeffs, flags=0)

output = {'success':success, 'r_vec':rotation_vector, 't_vec':translation_vector,'mtx':camera_matrix, 'dist':dist_coeffs}
pklname = os.path.join('data','2022_05_16',f'{cam_no}.pkl')
with open(pklname, 'wb') as handle:
    pickle.dump(output, handle)




# visualize
image_name = f'camera{cam_no}_calib.jpg'
fname = os.path.join('data','2022_05_16',image_name)
img = cv2.imread(fname)


proj_point2D, jacobian = cv2.projectPoints(carpet_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

error_2D = np.mean(np.sum((proj_point2D.reshape((-1,2))-carpet_2D)**2,axis=1)**0.5)
print(error_2D)

# plot
for p,p3 in zip(carpet_2D,proj_point2D):
    cv2.circle(img, (int(p[0]), int(p[1])), 15, (0,0,255), -1)
    cv2.circle(img, (int(p3[0][0]), int(p3[0][1])), 15, (0,255,0), 8)

# point1 = ( int(carpet_2D[0][0]), int(carpet_2D[0][1]))

# point2 = ( int(proj_point2D[0][0][0]), int(proj_point2D[0][0][1]))

# cv2.line(img, point1, point2, (255,255,255), 2)


# Display image
output_file = os.path.join('data','2022_05_16','output','correct.png')
cv2.imwrite(output_file,img)

# 14.167187160708364 for camera 8
# 17.41845113420259 for camera 1

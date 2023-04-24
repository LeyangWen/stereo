import os.path

import numpy as np
import cv2
import glob
import argparse
import sys
from calibration_store import load_coefficients, save_stereo_coefficients
import pickle
import pickle5


cam_no = str(1)

# load camera intrinsic extrinsic
pklname = os.path.join('data','2022_05_16',f'{cam_no}.pkl')
with open(pklname, 'rb') as handle:
    cam_info = pickle.load(handle)
print(cam_info['success'])

rotation_vector = cam_info['r_vec']
translation_vector = cam_info['t_vec']
camera_matrix = cam_info['mtx']
dist_coeffs = cam_info['dist']


# load image
image_name = f'camera{cam_no}_pose.jpg'
fname = os.path.join('data','2022_05_16',image_name)
img = cv2.imread(fname)

# load pose
rokoko_fname = os.path.join('data','2022_05_16','Rokoko_dict.pkl')
with open(rokoko_fname, 'rb') as handle:
    rokoko_raw = pickle5.load(handle)
pose3D_rel = rokoko_raw['pose_data'][0]*1000
print(rokoko_raw['keypoint_names'])
print(pose3D_rel[3])

# translate pose

width_mm = 223
length_mm = 330
x_translate = 3* width_mm
y_translate = 3* length_mm + pose3D_rel[3][1]
z_translate = 0

pose3D = pose3D_rel*np.array([-1,-1,1]) + np.array([x_translate,y_translate,z_translate])

# project pose


proj_point2D, jacobian = cv2.projectPoints(pose3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

# plot
for p3 in proj_point2D:
    # cv2.circle(img, (int(p[0]), int(p[1])), 10, (0,0,255), -1)
    cv2.circle(img, (int(p3[0][0]), int(p3[0][1])), 15, (0,255,0), 8)


# Display image
output_file = os.path.join('data','2022_05_16','output','output.png')
cv2.imwrite(output_file,img)
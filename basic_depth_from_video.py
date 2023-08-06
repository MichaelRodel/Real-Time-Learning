import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# from analysis.MotionArrowsOverlay import overlay_arrows_combined_frames
from depth_trackers.depth_analysis import depth_from_h264_vectors
from mapping.utils.Constants import Constants
from mapping.utils.file import load_camera_data_json

import open3d as o3d


image = np.full((200, 200, 3), 255, dtype=np.uint8)
cloud = np.empty((0, 3))

def show_cloud():
    global cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd])
    # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=4, std_ratio=0.5)
    inlier_cloud = cl.select_by_index(ind)
    o3d.visualization.draw_geometries([inlier_cloud])

def topdown_view(depth: np.ndarray, angle: float, max_dist: float = 1500):
    global image, cloud
    # depth map to cloud, clip it at 700cm to prevent outliers
    depth[:, 2] = np.clip(depth[:, 2], 0, max_dist)

    depth[:, :2] = np.squeeze(cv2.undistortPoints(depth[None, :, :2], cam_mat, dist_coeff))*depth[:, 2:]
    # depth is now a Nx3 3d point cloud
    rot_mat = Rotation.from_euler('y', angle, degrees=True).as_matrix()
    depth = depth @ rot_mat
    cloud = np.concatenate((cloud, depth))
    for point in depth:
        point = np.clip((point * 200 / max_dist + 100).astype(int), -200, 200)
        cv2.circle(image, (point[0], point[2]), 1, (0, 0, 0), -1)
    return image


save_video: bool = False  # turn off when saved video not required
show_video: bool = True  # turn off when video window not required
top_down: bool = True  # turn on when working on topdown view
show_depth_frame = True
cam_dir = os.path.join(Constants.ROOT_DIR, "mapping_wrappers/camera_config/pi/camera_data_480p.json")
cam_mat, dist_coeff, _ = load_camera_data_json(cam_dir)
path = os.path.join(Constants.ROOT_DIR, "results/depth_test1")
tello_angles = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
# angles1 = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
# angles2 = np.loadtxt(os.path.join(path, "tello_angles2.csv"))
# best_pair = generate_angle_pairs(angles1, angles2)  # doesn't work
cap1 = cv2.VideoCapture(os.path.join(path, "rot1.h264"))
cap2 = cv2.VideoCapture(os.path.join(path, "rot2.h264"))
if save_video:
    writer = cv2.VideoWriter(os.path.join(path, "depth.mp4"), cv2.VideoWriter_fourcc("M", "P", "4", "V"), 40,
                             (640, 480))
else:
    writer = None  # just to stop warning
# Initialize the feature detector (e.g., ORB, SIFT, etc.)
detector = cv2.ORB_create(nfeatures=11000)
# detector = cv2.xfeatures2d.SURF_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
for angle in tello_angles:
    # cap2.set(cv2.CAP_PROP_POS_FRAMES, pair - 1)  # seek to best pair, doesn't work
    ret1, frame1 = cap1.read()
    if not ret1:
        if save_video:
            writer.release()
        break
    ret2, frame2 = cap2.read()
    # combined_frame = np.concatenate((frame1, frame2), axis=1)
    # cv2.imshow("frames", combined_frame)
    # ORB
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    matches = [m for m, n in matches if m.distance < 0.50 * n.distance]
    print(len(matches), "matches using ORB")
    # show matches
    # frame_with_matches = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches[:100], None)
    # cv2.imshow("ORB matches", frame_with_matches)
    # cv2.waitKey(0)
    # continue
    # show depth

    ##############
    points1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
    points2 = np.array([keypoints2[match.trainIdx].pt for match in matches])
    # distances = np.array([match.distance for match in matches])
    # threshold = np.percentile(distances, 25)
    # points1 = np.array([keypoints1[match.queryIdx].pt for match in matches if match.distance < threshold])
    # points2 = np.array([keypoints2[match.trainIdx].pt for match in matches if match.distance < threshold])
    ##############
    depth = depth_from_h264_vectors(np.hstack((points1, points2)), cam_mat,
                                    30)  # you might want to save one of these for the topdown view
    if top_down:
        top_down_frame = topdown_view(np.hstack((points1, depth[:, None])), angle)
        if show_video:
            cv2.imshow("depth top down", top_down_frame)
    if show_depth_frame:
        depth_frame = frame1.copy()
        int_points1 = points1.astype(int)
        depth_color = np.clip(depth * 255 / 500, 0, 255)[:, None]  # clip  values from 0 to 5m and scale to 0-255(color range)
        for color, point in zip(depth_color, int_points1):
            cv2.rectangle(depth_frame, point[::] - 5, point[::] + 5, color, -1)
        if show_video:
            cv2.imshow("depth ORB", depth_frame)
            cv2.waitKey(1)  # need some minimum time because opencv doesnt work without it
    if save_video:
        writer.write(depth_frame)
show_cloud()
input()
# similar method that uses motion vectors
# points3d = triangulate_points(keypoints1, keypoints2, matches, 60, cam_mat, dist_coeff)
# depth_frame = frame1.copy()
# frame1[(points3d[:, :2]).astype(int)] = 255*(points3d[:, 3]/np.max(points3d[:, 3]))

# Motion Vectors
# vectors = ffmpeg_encode_extract(frame1, frame2, subpixel=False)
# print(len(vectors), "matches using MV")
# combined_frame = np.concatenate((frame1, frame2), axis=1)
# combined_frame = overlay_arrows_combined_frames(combined_frame, vectors, max_vectors=50)
# cv2.imshow("MV matches", combined_frame)
# cv2.waitKey(0)

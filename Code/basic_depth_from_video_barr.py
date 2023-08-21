import os
from itertools import repeat

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
import mapping.utils.raspberrypi.picamerautil
import mapping.utils.vector_conversions
from angle_trackers.angle_estimators import angle_diff_tan
from angle_trackers.pi_vector_tracker import PiVectorTracker
from mvextractor.videocap import VideoCap
from mapping.utils.vector_conversions import motion_vectors_conversion
# from analysis.MotionArrowsOverlay import overlay_arrows_combined_frames
from depth_trackers.depth_analysis import depth_from_h264_vectors, generate_angle_pairs
from mapping.utils.Constants import Constants
from mapping.utils.file import load_camera_data_json
from mapping.utils.motionVectors.mv_extractor.ffmpeg_enc import ffmpeg_encode_extract
from analysis.MotionArrowsOverlay import overlay_arrows_combined_frames
from mapping.utils.raspberrypi.picamerautil import create_masks_horizontal, create_masks_vertical

cloud = o3d.geometry.PointCloud()
# cloud = np.empty((0, 3))
image = None


def topdown_view(depth: np.ndarray, angle: float, max_dist: float = 1500, keep_image: bool = True):
    global image, cloud
    if not keep_image or image is None:
        image = np.full((300, 300, 3), 255, dtype=np.uint8)
    # depth map to cloud, clip it at max_dist to prevent extremely far outliers
    # depth[:, 2] = np.clip(depth[:, 2], 0, max_dist)

    depth[:, :2] = -np.squeeze(cv2.undistortPoints(depth[None, :, :2], cam_mat, dist_coeff)) * depth[:, 2:]
    # depth is now a Nx3 3d point cloud
    rot_mat = Rotation.from_euler('y', angle, degrees=True).as_matrix()
    depth = depth @ rot_mat
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depth)
    clean_cloud, indices = pcd.remove_statistical_outlier(nb_neighbors=32, std_ratio=2)
    cloud += clean_cloud
    # o3d.visualization.draw_geometries([clean_cloud])
    # cloud = np.concatenate((cloud, depth))
    for point in depth:
        new_point = np.clip((point * 300 / max_dist + 150).astype(int), 0, 300)
        y_val = np.clip(point[1], 0, 255)
        cv2.circle(image, (new_point[0], new_point[2]), 1, (y_val, y_val, y_val), -1)
    return image


def show_cloud_pcd():
    # clean_cloud, indices = cloud.remove_statistical_outlier(nb_neighbors=64, std_ratio=1)
    o3d.visualization.draw_geometries([cloud])


def show_cloud():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    to_show = np.random.choice(cloud.shape[0], 1000, replace=True)
    ax.scatter(cloud[to_show, 0], cloud[to_show, 1], cloud[to_show, 2])
    ax.set_box_aspect((np.ptp(cloud[to_show, 0]), np.ptp(cloud[to_show, 1]), np.ptp(cloud[to_show, 2])))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(90, 0, 0)
    plt.show()


save_video: bool = False  # turn off when saved video not required
show_video: bool = True  # turn off when video window not required
top_down: bool = True  # turn on when working on topdown view
show_depth_frame: bool = True
use_mv: bool = False
print_count = True
bottom = True
height_filter: bool = True
wait = 0

# parameters to change:
max_height_over = -10  # cm to keep above drone, negative because bigger y is higher
max_height_below = 100  # cm to keep below drone
ratio_test = 0.9  # parameter for ratio test
percentile = 30  # parameter for percentile filter
height_diff = 30  # distance between drone locations

cam_dir = os.path.join(Constants.ROOT_DIR, "mapping_wrappers/camera_config/pi/camera_data_480p.json")
cam_mat, dist_coeff, _ = load_camera_data_json(cam_dir)
path = os.path.join(Constants.ROOT_DIR, "results/depth_test1")
tello_angles_low = np.loadtxt(os.path.join(path, "tello_angles_low.csv"))
tello_angles_high = np.loadtxt(os.path.join(path, "tello_angles_high.csv"))


# tello_heights = np.loadtxt(os.path.join(path, "tello_heights.csv"))


def load_frames_and_angles(path):
    cap = VideoCap()
    cap.open(path)
    frames = []
    angles = []
    tracker = PiVectorTracker(angle_diff_tan, cam_mat, dist_coeff, median=True, cumulative=True)
    for _ in tqdm(repeat(1)):
        ret, frame, extracted_mvs, frame_type, timestamp = cap.read()
        if ret:
            frames.append(frame)
            angles.append(tracker.track_frame(motion_vectors_conversion(extracted_mvs)))
        else:
            return frames


def load_frames(cap):
    frames = []
    for _ in tqdm(repeat(1)):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            return frames


cap1 = cv2.VideoCapture(os.path.join(path, "rot_low.h264"))
frames_low = load_frames(cap1)
vectors_low = [mapping.utils.vector_conversions.pi_vectors_conversion(mv) for mv in
               np.load(os.path.join(path, "motion_data_low.npy")).reshape((-1, 30, 41))]
# tracker = PiVectorTracker(angle_diff_tan, cam_mat, dist_coeff, median=True, cumulative=True)
# angles_low = np.array(tracker.track_frame_batch(vectors_low))+tello_angles_low[0]
# angles1 = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
# angles2 = np.loadtxt(os.path.join(path, "tello_angles2.csv"))
cap2 = cv2.VideoCapture(os.path.join(path, "rot_high.h264"))
frames_high = load_frames(cap2)
# vectors_high = [mapping.utils.vector_conversions.pi_vectors_conversion(mv) for mv in
#                 np.load(os.path.join(path, "motion_data_high.npy")).reshape((-1, 30, 41))]
# tracker = PiVectorTracker(angle_diff_tan, cam_mat, dist_coeff, median=True, cumulative=True)
# angles_high = np.array(tracker.track_frame_batch(vectors_high))+tello_angles_high[0]
if save_video:
    writer = cv2.VideoWriter(os.path.join(path, "depth.mp4"), -1, 40, (640, 480))
else:
    writer = None  # just to stop warning
# Initialize the feature detector (e.g., ORB, SIFT, etc.)
detector = cv2.ORB_create()
# alignment = generate_angle_pairs(angles_high, angles_low)
alignment = generate_angle_pairs(tello_angles_high, tello_angles_low)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
for i, (frame_high, angle) in enumerate(zip(frames_high, tello_angles_high)):
    # for i, (frame_high, angle) in enumerate(zip(frames_high, angles_high)):
    frame1 = frames_low[alignment[i]]
    frame2 = frame_high
    # Motion Vectors
    if use_mv:  # motion vectors vs ORB
        vectors = ffmpeg_encode_extract(frame1, frame2, subpixel=True)
        if print_count:
            print(len(vectors), "matches using MV")
        big_enough = (vectors[:, 3] - vectors[:, 1]) > 1
        vectors = vectors[big_enough]
        depth = depth_from_h264_vectors(vectors, cam_mat, 30)
        if height_filter:
            high_enough = ((vectors[:, 3] - cam_mat[1, 2]) / cam_mat[1, 1] * depth) < 100
            low_enough = ((vectors[:, 3] - cam_mat[1, 2]) / cam_mat[1, 1] * depth) > -50
            vectors = vectors[low_enough & high_enough, :]
            depth = depth[low_enough & high_enough]
        if print_count:
            print(len(vectors), "matches using MV after filter")
        int_vecs = vectors.astype(int)
        combined_frame = np.concatenate((frame1, frame2), axis=1)
        combined_frame = overlay_arrows_combined_frames(combined_frame, int_vecs, max_vectors=50)
        cv2.imshow("MV matches", combined_frame)
        points1 = vectors[:, 2:]
        int_points1 = int_vecs[:, 2:]
    else:
        # ORB
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
        # Perform the matching
        # matches = matcher.match(descriptors1, descriptors2)
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        matches = [m for m, n in matches if m.distance < ratio_test * n.distance]
        if not matches:
            continue
        if print_count:
            print(len(matches), "matches using ORB")
        # show matches
        frame_with_matches = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches, None)
        cv2.imshow("ORB matches", frame_with_matches)
        # show depth
        distances = np.array([match.distance for match in matches])
        threshold = np.percentile(distances, percentile)
        # threshold = 30#np.mean(distances)
        # print(threshold)
        points1 = np.array([keypoints1[match.queryIdx].pt for match in matches if match.distance < threshold])
        points2 = np.array([keypoints2[match.trainIdx].pt for match in matches if match.distance < threshold])
        if len(points1) <= 1:
            continue
        depth = depth_from_h264_vectors(np.hstack((points1, points2)), cam_mat, height_diff)
        if not bottom:
            points1 = points2
        if height_filter:
            high_enough = ((points1[:, 1] - cam_mat[1, 2]) / cam_mat[1, 1] * depth) < max_height_below - (height_diff if bottom else 0)
            low_enough = ((points1[:, 1] - cam_mat[1, 2]) / cam_mat[1, 1] * depth) > max_height_over - (height_diff if bottom else 0)
            points1 = points1[low_enough & high_enough, :]
            depth = depth[low_enough & high_enough]
        # TODO implement height filter like in MV
        int_points1 = points1.astype(int)
    if top_down and len(depth) != 0:
        top_down_frame = topdown_view(np.hstack((points1, depth[:, None])), angle)
        top_down_frame = top_down_frame.copy()
        cv2.line(top_down_frame, (150, 150), (150 + (300 * np.cos(np.deg2rad(angle + 90))).astype(int),
                                              150 + (300 * np.sin(np.deg2rad(angle + 90))).astype(int)), (255, 0, 0))
        if show_video:
            cv2.imshow("depth top down", top_down_frame)
    if show_depth_frame:
        depth_frame = (frame1 if bottom else frame2).copy()
        depth_color = np.clip(depth * 255 / 1000, 0, 255)[:,
                      None]  # clip  values from 0 to 10m and scale to 0-255(color range)
        for color, point in zip(depth_color, int_points1):
            cv2.rectangle(depth_frame, point - 5, point + 5, color, -1)
        if show_video:
            cv2.imshow(f"depth {'MV' if use_mv else 'ORB'}", depth_frame)
    cv2.waitKey(wait)  # need some minimum time because opencv doesnt work without it
    if save_video:
        writer.write(depth_frame)
show_cloud_pcd()
# input()
# similar method that uses motion vectors
# points3d = triangulate_points(keypoints1, keypoints2, matches, 60, cam_mat, dist_coeff)
# depth_frame = frame1.copy()
# frame1[(points3d[:, :2]).astype(int)] = 255*(points3d[:, 3]/np.max(points3d[:, 3]))

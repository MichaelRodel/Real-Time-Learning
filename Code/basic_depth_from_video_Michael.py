import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
# from analysis.MotionArrowsOverlay import overlay_arrows_combined_frames
from depth_trackers.depth_analysis import depth_from_h264_vectors
from mapping.utils.Constants import Constants
from mapping.utils.file import load_camera_data_json

import open3d as o3d

cloud = np.empty((0, 3))
image = np.full((200, 200, 3), 255, dtype=np.uint8)

# new start
depth_cloud = None


def cart3D2pol3D(cart3D_cloud):
    # TODO: check this function
    x = cart3D_cloud[, :0]
    y = cart3D_cloud[, :1]
    z = cart3D_cloud[, :2]

    x_2 = x ** 2
    y_2 = y ** 2
    z_2 = z ** 2

    xy = np.sqrt(x_2 + y_2)

    r = np.sqrt(x_2 + y_2 + z_2)

    theta = np.arctan2(y, x)  # TODO: write the plane for the angle
    phi = np.arctan2(xy, z)  # TODO: write the plane for the angle

    pol3D = [r, theta, phi]
    return pol3D


def pol3D2cart3D(pol3D_cloud):
    # TODO: check this function
    r = pol3D_cloud[, :0]
    theta = pol3D_cloud[, :1]
    phi = pol3D_cloud[, :2]
    cart3D = [r * np.sin(theta) * np.cos(phi),
              r * np.sin(theta) * np.sin(phi),
              r * np.cos(theta)
              ]
    return cart3D


def clear_cloud(pcd):
    # algo 1: isolated outlier removal

    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])


# new end

def topdown_view(depth: np.ndarray, angle: float, max_dist: float = 1500):
    global image, cloud
    # depth map to cloud, clip it at max_dist to prevent extremely far outliers
    depth[:, 2] = np.clip(depth[:, 2], 0, max_dist)

    depth[:, :2] = -np.squeeze(cv2.undistortPoints(depth[None, :, :2], cam_mat, dist_coeff)) * depth[:, 2:]
    # depth is now a Nx3 3d point cloud
    rot_mat = Rotation.from_euler('y', angle, degrees=True).as_matrix()
    depth = depth @ rot_mat
    cloud = np.concatenate((cloud, depth))
    for point in depth:
        point = np.clip((point * 200 / max_dist + 100).astype(int), -200, 200)
        cv2.circle(image, (point[0], point[2]), 1, (0, 0, 0), -1)
    return image


def show_cloud():
    global cloud

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd])

    # new begin
    clear_cloud(pcd)
    # new end

    # detect outliers
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=4, std_ratio=0.8)
    inlier_cloud = cl.select_by_index(ind)
    o3d.visualization.draw_geometries([inlier_cloud])
    # Radius outlier removal
    # cl, ind = pcd.remove_radius_outlier(nb_points=4, radius=0.09)
    # inlier_cloud = cl.select_by_index(ind)
    # o3d.visualization.draw_geometries([inlier_cloud])

    # boundary detection
    # boundarys, mask = pcd.get_axis_aligned_bounding_box()
    # boundarys = boundarys.paint_uniform_color([1.0, 0.0, 0.0])
    # pcd = pcd.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([pcd.to_legacy(), boundarys.to_legacy()],
    #                                   zoom=0.3412,
    #                                   front=[0.3257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    # new end

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # to_show = np.random.choice(cloud.shape[0], 1000, replace=True)
    # ax.scatter(cloud[to_show, 0], cloud[to_show, 1], cloud[to_show, 2])
    # ax.set_box_aspect((np.ptp(cloud[to_show, 0]), np.ptp(cloud[to_show, 1]), np.ptp(cloud[to_show, 2])))
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()


save_video: bool = False  # turn off when saved video not required
show_video: bool = True  # turn off when video window not required
top_down: bool = True  # turn on when working on topdown view
show_depth_frame: bool = True
cam_dir = os.path.join(Constants.ROOT_DIR, "mapping_wrappers/camera_config/pi/camera_data_480p.json")
cam_mat, dist_coeff, _ = load_camera_data_json(cam_dir)
path = os.path.join(Constants.ROOT_DIR, "results/depth_test1")
tello_angles = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
# vectors = [mapping.utils.vector_conversions.pi_vectors_conversion(mv).astype(int) for mv in
#            np.load(os.path.join(path, "motion_data1.npy")).reshape((-1, 30, 41))]
# angles1 = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
# angles2 = np.loadtxt(os.path.join(path, "tello_angles2.csv"))
# best_pair = generate_angle_pairs(angles1, angles2)  # doesn't work
cap1 = cv2.VideoCapture(os.path.join(path, "rot1.h264"))
cap2 = cv2.VideoCapture(os.path.join(path, "rot2.h264"))
if save_video:
    writer = cv2.VideoWriter(os.path.join(path, "depth.mp4"), -1, 40, (640, 480))
else:
    writer = None  # just to stop warning
# Initialize the feature detector (e.g., ORB, SIFT, etc.)
detector = cv2.ORB_create(nfeatures=11000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

for angle in tello_angles:
    # if angle % 60 == 0:
    print(f"{angle=}")
    # cap2.set(cv2.CAP_PROP_POS_FRAMES, pair - 1)  # seek to best pair, doesn't work
    ret1, frame1 = cap1.read()
    # frame1[len(frame1)-1] = 0
    # cv2.imshow('frame1', frame1)
    # cv2.waitKey(1)

    if not ret1:
        if save_video:
            writer.release()
        break
    ret2, frame2 = cap2.read()
    # cv2.imshow('frame2', frame2)
    # cv2.waitKey(1)

    # combined_frame = np.concatenate((frame1, frame2), axis=1)
    # cv2.imshow("frames", combined_frame)
    # Motion Vectors
    # vectors = ffmpeg_encode_extract(frame1, frame2, subpixel=False)
    # print(len(vectors), "matches using MV")
    # combined_frame = np.concatenate((frame1, frame2), axis=1)
    # combined_frame = overlay_arrows_combined_frames(combined_frame, vectors, max_vectors=50)
    # cv2.imshow("MV matches", combined_frame)
    # cv2.waitKey(0)
    # continue
    # ORB
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and compute descriptors for both images mask1, mask2 = create_masks_vertical(frame1, vectors,
    # frame2, False)   # Use this line instead of the one above when not showing image mask1 =
    # create_masks_horizontal(frame1, frame_vectors, False)   # Use this line instead of the one above when not
    # showing image cv2.imshow("mask", mask1)
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    # Perform the matching
    # matches = matcher.match(descriptors1, descriptors2)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    matches = [m for m, n in matches if m.distance < 0.9 * n.distance]
    print(len(matches), "matches using ORB")
    # show matches

    # frame_with_matches = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches[:100], None)
    # cv2.imshow("ORB matches", frame_with_matches)
    # cv2.waitKey(0)
    # continue
    # show depth
    distances = np.array([match.distance for match in matches])
    threshold = np.percentile(distances, 0.75)
    points1 = np.array([keypoints1[match.queryIdx].pt for match in matches if match.distance < threshold])
    points2 = np.array([keypoints2[match.trainIdx].pt for match in matches if match.distance < threshold])
    depth = depth_from_h264_vectors(np.hstack((points1, points2)), cam_mat, 30)
    if top_down and len(depth) != 0:
        top_down_frame = topdown_view(np.hstack((points1, depth[:, None])), angle)

        depth_cloud = top_down_frame

        if show_video:
            cv2.imshow("cloud depth top down", top_down_frame)
            cv2.waitKey(1)

    if show_depth_frame:
        depth_frame = frame1.copy()
        int_points1 = points1.astype(int)
        depth_color = np.clip(depth * 255 / 1000, 0, 255)[:,
                      None]  # clip  values from 0 to 10m and scale to 0-255(color range)
        for color, point in zip(depth_color, int_points1):
            cv2.rectangle(depth_frame, point - 5, point + 5, color, -1)
        if show_video:
            cv2.imshow("depth ORB", depth_frame)
            cv2.waitKey(1)  # need some minimum time because opencv doesn't work without it
        if save_video:
            writer.write(depth_frame)
# input()
show_cloud()

# similar method that uses motion vectors
# points3d = triangulate_points(keypoints1, keypoints2, matches, 60, cam_mat, dist_coeff)
# depth_frame = frame1.copy()
# frame1[(points3d[:, :2]).astype(int)] = 255*(points3d[:, 3]/np.max(points3d[:, 3]))

import os

import cv2
import numpy
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
# from analysis.MotionArrowsOverlay import overlay_arrows_combined_frames
from depth_trackers.depth_analysis import depth_from_h264_vectors
from mapping.utils.Constants import Constants
from mapping.utils.file import load_camera_data_json

import open3d as o3d

# from sklearn.decomposition import PCA  # new
#
# pca = PCA(n_components=3)

cloud = np.empty((0, 3))
image = np.full((200, 200, 3), 255, dtype=np.uint8)

# new start
depth_cloud = None


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
    # new begin
    # concat_1 = cv2.hconcat(images_list_1)
    # concat_2 = cv2.hconcat(images_list_2)
    # cv2.imshow('concat1', concat_1)
    # # cv2.waitKey(0)
    # cv2.imshow('concat2', concat_2)
    # cv2.waitKey(0)

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

# images_list_1 = []
# images_list_2 = []

for angle in tello_angles:
    # if angle % 60 == 0:
    print(f"{angle=}")
    # cap2.set(cv2.CAP_PROP_POS_FRAMES, pair - 1)  # seek to best pair, doesn't work
    ret1, frame1 = cap1.read()
    # frame1[len(frame1)-1] = 0
    # cv2.imshow('frame1', frame1)
    # cv2.waitKey(1)

    # images_list_1.append(frame1)

    if not ret1:
        if save_video:
            writer.release()
        break
    ret2, frame2 = cap2.read()
    # frame2[len(frame2) - 1] = 0
    # cv2.imshow('frame2', frame2)
    # cv2.waitKey(1)

    # images_list_2.append(frame2)

    # concat_1 = cv2.hconcat(images_list_1)
    # concat_2 = cv2.hconcat(images_list_2)
    # cv2.imshow('concat1', concat_1)
    # cv2.waitKey(0)
    # cv2.imshow('concat2', concat_2)
    # cv2.waitKey(0)

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

            # Apply edh_frame_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
            #             # edges = cv2.Canny(depth_frame_gray, 50, 150, apertureSize=3)
            #             #
            #             # # This returns an array of r and theta values
            #             # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            #             #
            #             # # The below for loop runs till r and theta values
            #             # # are in the range of the 2d array
            #             # if lines is not None:
            #             #     for r_theta in lines:
            #             #         arr = np.array(r_theta[0], dtype=np.float64)
            #             #         r, theta = arr
            #             #         # Stores the value of cos(theta) in a
            #             #         a = np.cos(theta)
            #             #
            #             #         # Stores the value of sin(theta) in b
            #             #         b = np.sin(theta)
            #             #
            #             #         # x0 stores the value rcos(theta)
            #             #         x0 = a * r
            #             #
            #             #         # y0 stores the value rsin(theta)
            #             #         y0 = b * r
            #             #
            #             #         # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            #             #         x1 = int(x0 + 1000 * (-b))
            #             #
            #             #         # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            #             #         y1 = int(y0 + 1000 * (a))
            #             #
            #             #         # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            #             #         x2 = int(x0 - 1000 * (-b))
            #             #
            #             #         # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            #             #         y2 = int(y0 - 1000 * (a))
            #             #
            #             #         # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            #             #         # (0,0,255) denotes the colour of the line to be
            #             #         # drawn. In this case, it is red.
            #             #         cv2.line(depth_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #             #         cv2.imshow("lines and corners", depth_frame)
            #             #         cv2.waitKey(1)ge detection method on the image
            # dept

            # t_lower = 100  # Lower Threshold
            # t_upper = 200  # Upper threshold
            # aperture_size = 5  # Aperture size
            # L2Gradient = True  # Boolean
            #
            # # Applying the Canny Edge filter with L2Gradient = True
            # edge = cv2.Canny(depth_frame, t_lower, t_upper, L2gradient=L2Gradient)
            # cv2.imshow('edge', edge)
            # cv2.waitKey(1)
            # end new

        if save_video:
            writer.write(depth_frame)
# input()
show_cloud()

# similar method that uses motion vectors
# points3d = triangulate_points(keypoints1, keypoints2, matches, 60, cam_mat, dist_coeff)
# depth_frame = frame1.copy()
# frame1[(points3d[:, :2]).astype(int)] = 255*(points3d[:, 3]/np.max(points3d[:, 3]))

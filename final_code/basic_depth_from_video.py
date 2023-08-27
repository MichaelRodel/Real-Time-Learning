import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# from analysis.MotionArrowsOverlay import overlay_arrows_combined_frames
from depth_trackers.depth_analysis import depth_from_h264_vectors
from mapping.utils.Constants import Constants
from mapping.utils.file import load_camera_data_json

import open3d as o3d
import time


image = np.full((200, 200, 3), 255, dtype=np.uint8)  # top-down 2d image
final_image = np.copy(image)
cloud = np.empty((0, 3))                             # our 3D cloud

def show_cloud(cloud):                                  # partial inspiration from https://stackoverflow.com/questions/62948421/how-to-create-point-cloud-file-ply-from-vertices-stored-as-numpy-array
    """
    Function for visualizing point cloud, using open3d library.
    Detects if cloud is 2D and supports that case as well.
    """
    if cloud.shape[1] == 2:
        cloud = np.append(np.zeros((cloud.shape[0], 1)), cloud, axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd])

def clean_outliers_statistical(cloud):                   # inspiration from http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
    """
    Function for removing outliers using statistical outlier removal function of open3d library.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=0.7)
    # inlier_cloud = cl.select_by_index(ind)
    return np.asarray(cl.points)


def cart2D2pol2D(cart2D_cloud):                   # partial inspiration from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    """
    Function for converting 2D cloud point from cartesian representation to polar representation.
    """
    x_2 = np.power(cart2D_cloud[:, 0], 2)
    y_2 = np.power(cart2D_cloud[:, 1], 2)
    rho = np.sqrt(np.add(x_2, y_2))
    phi = np.arctan2(cart2D_cloud[:, 1], cart2D_cloud[:, 0])
    new_cloud = np.array([rho, phi])
    return new_cloud.T


def pol2D2cart2D(pol2D_cloud):                    # partial inspiration from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    """
    Function for converting 2D cloud point back from polar representation to cartesian representation.
    """
    x = np.multiply(pol2D_cloud[:, 0], np.cos(pol2D_cloud[:, 1]))
    y = np.multiply(pol2D_cloud[:, 0], np.sin(pol2D_cloud[:, 1]))
    new_cloud = np.array([x, y])
    return new_cloud.T


def cart3D2pol3D(cart3D_cloud):  # inspiration from https://stackoverflow.com/questions/68308053/how-to-manipulate-a-3d-array-to-convert-from-cartesian-coordinates-to-spherical
    """
    Function for converting 3D cloud point from cartesian representation to polar/sphere representation.
    """
    x = cart3D_cloud[:, 0]
    y = cart3D_cloud[:, 1]
    z = cart3D_cloud[:, 2]

    x_2 = x ** 2
    y_2 = y ** 2
    z_2 = z ** 2

    xy = np.sqrt(x_2 + y_2)

    r = np.sqrt(x_2 + y_2 + z_2)

    theta = np.arctan2(y, x)
    phi = np.arctan2(xy, z)

    return np.array([r, theta, phi]).T


def pol3D2cart3D(pol3D_cloud):     # inspiration from https://stackoverflow.com/questions/48348953/spherical-polar-co-ordinate-to-cartesian-co-ordinate-conversion
    """
    Function for converting 3D cloud point back from sphere representation to cartesian representation.
    """
    r = pol3D_cloud[:, 0]
    theta = pol3D_cloud[:, 1]
    phi = pol3D_cloud[:, 2]
    cart3D = np.array([r * np.cos(theta) * np.sin(phi),
              r * np.sin(theta) * np.sin(phi),
              r * np.cos(phi)
              ])
    cart3D = np.array(np.vsplit(np.transpose(cart3D), cart3D.shape[1]))
    cart3D = np.array([c[0] for c in cart3D])
    return cart3D

def topdown_view(depth: np.ndarray, angle: float, max_dist: float = 1500):            # originally implemented by us, later replaced by different implementation from Barr
    """
    Function that calculates the 3D + top-down clouds for each pair of frames.
    """
    global image, cloud
    # depth map to cloud, clip it at 700cm to prevent outliers
    depth[:, 2] = np.clip(depth[:, 2], 0, max_dist)

    depth[:, :2] = - np.squeeze(cv2.undistortPoints(depth[None, :, :2], cam_mat, dist_coeff))*depth[:, 2:]
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
detector = cv2.ORB_create(nfeatures=11000)                       # ORB detector
# detector = cv2.xfeatures2d.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)      # Brute Force matcher
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
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)          # key points + descriptors for lower frame
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)          # key points + descriptors for upper frame
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)                # match
    matches = [m for m, n in matches if m.distance < 0.50 * n.distance]        # apply ratio test
    print(len(matches), "matches using ORB")


    ##############
    points1 = np.array([keypoints1[match.queryIdx].pt for match in matches])   # points from below
    points2 = np.array([keypoints2[match.trainIdx].pt for match in matches])   # points from above
    ##############
    depth = depth_from_h264_vectors(np.hstack((points1, points2)), cam_mat,
                                    30)  # you might want to save one of these for the topdown view
    if top_down:
        top_down_frame = topdown_view(np.hstack((points1, depth[:, None])), angle)
        final_image = np.copy(top_down_frame)
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
# cloud = np.array([cl in cloud if -50 <= cl[1] <= 100])
show_cloud(cloud=cloud)

# original
def organize_local(point1, point2, points: np.ndarray):
    """
    Given two boundary points for the square and the points within it, compute new point in the center of the square
    with R value taken as the 20-th percentile of all the R values of the points within the square.
    """
    if points.shape[0] == 0:
        return None
    new_point = np.zeros(3)
    new_point[1] = (point1[1] + point2[1]) / 2
    new_point[2] = (point1[2] + point2[2]) / 2
    # new_point[0] = np.mean(points[:, 0])
    new_point[0] = np.percentile(points[:, 0], 20)
    return new_point

# original
def organize_local2d(point1, point2, points: np.ndarray):
    """
    Same as above function but fro 2D.
    """
    if points.shape[0] == 0:
        return None
    new_point = np.zeros(2)
    new_point[1] = (point1[1] + point2[1]) / 2
    new_point[0] = np.percentile(points[:, 0], 10)
    return new_point

# original
def cloud_filter(cloud, use3D=True, jump_size=None, window_size=None, angle_window=(-180, 180)):     # general idea for algorithm provided to us by Barr
    """
    our Final filtering algorithm.
    """
    assert cloud.shape[1] == 3
    if use3D:
        if jump_size is None:
            jump_size = 3
        if window_size is None:
            window_size = 5
        assert jump_size <= window_size
        cloud = cart3D2pol3D(cloud)
        cloud[:, [1, 2]] = np.rad2deg(cloud[:, [1, 2]])  # convert radians to degrees
        show_cloud(cloud=cloud)
        start_time = time.time()
        cloud1 = []
        # cloud = cloud[np.apply_along_axis(lambda row: row[2], 1, cloud).argsort()]
        for i in range(angle_window[0], angle_window[1], jump_size):                  # iterate through squares
            for j in range(0, 180, jump_size):
                # below is same as: points = np.array([p for p in cloud if i <= p[1] <= i + window_size and j <= p[2] <= j + window_size])
                filter = (cloud[:, 1] <= i+window_size) & (cloud[:, 1] >= i)
                filter &= (cloud[:, 2] <= j+window_size)
                filter &= (cloud[:, 2] >= j)
                points = cloud[filter]
                ret = organize_local(np.array([0, i, j]), np.array([0, i+window_size, j+window_size]), points)
                if ret is not None:
                    cloud1.append(ret)         # add new points to the cloud
        cloud = np.array(cloud1)               # update cloud
        cloud[:, [1, 2]] = np.deg2rad(cloud[:, [1, 2]])           # convert radians to degrees
        cloud = pol3D2cart3D(cloud)            # back to cartesian representation
        print(f"took {time.time() - start_time} seconds.")
        show_cloud(cloud=cloud)
        cloud = clean_outliers_statistical(cloud) # statistical outlier removal
        show_cloud(cloud=cloud)
        return cloud
    else:
        ############################
        # same as above but for 2D.
        ############################
        cloud = cloud[:, [0, 2]]
        cloud = cart2D2pol2D(cloud)
        cloud[:, 1] = np.rad2deg(cloud[:, 1])  # convert radians to degrees
        show_cloud(cloud=cloud)
        if window_size is None:
            window_size = 2  # in degrees
        if jump_size is None:
            jump_size = 1
        assert jump_size <= window_size
        cloud1 = []
        for i in range(angle_window[0], angle_window[1], jump_size):
            points = cloud[(cloud[:, 1] <= i+window_size) & (cloud[:, 1] >= i)]
            ret = organize_local2d(np.array([0, i]), np.array([0, i + window_size]), points)
            if ret is not None:
                cloud1.append(ret)
        cloud = np.array(cloud1)
        cloud[:, 1] = np.deg2rad(cloud[:, 1])  # convert radians to degrees
        cloud = pol2D2cart2D(cloud)
        show_cloud(cloud=cloud)
        return cloud

c = cloud_filter(cloud, use3D=True)          # apply our filter
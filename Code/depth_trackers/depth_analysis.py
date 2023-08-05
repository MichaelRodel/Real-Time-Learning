import cv2
import numpy as np


# TODO can be a lot faster
def generate_angle_pairs(angles1: np.ndarray, angles2: np.ndarray):
    """
    Finds the closest angle from angles2 for each angle in angles1
    """
    return np.argmax((angles1[:, None] - (angles2 % 360)), axis=1)


def depth_from_h264_vectors(vectors: np.ndarray, camera_matrix: np.ndarray, diff: float):
    """
    Estimates depth of each 3d point, represented as a vector(x1, y1, x2, y2) between its
    2d location in 2 consecutive frames with a known height difference
    """
    assert diff != 0
    if len(vectors) == 0:
        return np.empty(0)
    with np.errstate(divide='ignore'):
        return diff * camera_matrix[1, 1] / np.abs(vectors[:, 3] - vectors[:, 1])


def depth_from_pi_vectors(vectors: np.ndarray, camera_matrix: np.ndarray, diff, **kwargs):
    """
    Estimates depth of each 3d point, represented in
     a picamera motion vectors matrix(MxN matrix with (x_diff, y_diff, SAD) in each cell)
     between 2 consecutive frames with a known height difference
    """
    return diff * camera_matrix[1, 2] / np.abs(vectors["x"])


def combine_depths(depth1: np.ndarray, depth2: np.ndarray, angle_diff):
    """
     Combines 2 depth2 that can be overlapping
    """


def triangulate_points(keypoints1, keypoints2, matches, diff, cam_mat, distortion_coeff):
    projMat1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # origin
    projMat2 = np.hstack([np.eye(3), np.array((0, diff, 0))[:, None]])  # R, T
    points1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
    points2 = np.array([keypoints2[match.trainIdx].pt for match in matches])
    points1u = cv2.undistortPoints(points1, cam_mat, distortion_coeff)
    points2u = cv2.undistortPoints(points2, cam_mat, distortion_coeff)
    points4d = cv2.triangulatePoints(projMat1, projMat2, points1u, points2u)
    points3d = (points4d[:3, :] / points4d[3, :]).T
    return points3d


if __name__ == "__main__":
    import os.path
    from mapping.utils.Constants import Constants
    path = os.path.join(Constants.ROOT_DIR, "results/depth_test1")
    angles1 = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
    angles2 = np.loadtxt(os.path.join(path, "tello_angles2.csv"))
    best_pair = generate_angle_pairs(angles1, angles2)
    cap1 = cv2.VideoCapture(os.path.join(path, "rot1.h264"))
    cap2 = cv2.VideoCapture(os.path.join(path, "rot2.h264"))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, best_pair[0] - 1)  # seek to best pair
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    combined_frame = np.concatenate((frame1, frame2), axis=1)
    cv2.imshow("frames", combined_frame)
    cv2.waitKey(0)

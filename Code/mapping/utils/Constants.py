from cv2 import ORB_create, ORB_FAST_SCORE
from numpy import pi
from pathlib import Path


class Constants:
	FLANN = False  # FLANN seems to be around 30% faster, accuracy to be tested
	USE_BETTER_VECTORS = False
	N_FEATURES = 2000
	ANGLE_THRESHOLD = pi / 5
	GOOD_RATIO = 0.65
	DESCRIPTOR_CONSTRUCTOR = ORB_create  # SIFT_create
	FAST_THRESHOLD = 10
	FEATURES_SCORE = ORB_FAST_SCORE
	DESCRIPTOR_ARGS = {
		'nfeatures': N_FEATURES,
		'scoreType': FEATURES_SCORE,
		'fastThreshold': FAST_THRESHOLD
	} if DESCRIPTOR_CONSTRUCTOR == ORB_create else {
		'nfeatures': N_FEATURES
	}
	DESCRIPTOR_OBJECT = DESCRIPTOR_CONSTRUCTOR(**DESCRIPTOR_ARGS)

	LITEMAP_HEIGHT_DIFFERENCE = 30  # cm

	MERGE_DISTANCE_THRESHOLD = 5
	MERGE_NEW_POINTS = False
	REMOVE_MERGE_DUPLICATES = False
	MAX_DISTANCE_ANGLE_MAPPER = 75
	MEMOIZATION_FACTOR = 0.5

	# TELLO PARAMETERS
	TELLO_FOV = 62
	TELLO_SSID = "AAP"
	TELLO_IP = "192.168.13.222"
	IsRaspberry = True
	ROOT_DIR = Path(__file__).parent.parent.parent

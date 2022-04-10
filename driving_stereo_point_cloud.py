import cv2
import numpy as np
import glob
import open3d as o3d

from crestereo import CREStereo, CameraConfig
from crestereo.utils import Open3dVisualizer

def get_driving_stereo_images(base_path, start_sample=0):

	# Get image list
	left_images = glob.glob(f'{base_path}/left/*.jpg')
	left_images.sort()
	right_images = glob.glob(f'{base_path}/right/*.jpg')
	right_images.sort()
	depth_images = glob.glob(f'{base_path}/depth/*.png')
	depth_images.sort()

	return left_images[start_sample:], right_images[start_sample:], depth_images[start_sample:]

# Model options (not all options supported together)
iters = 5            # Lower iterations are faster, but will lower detail. 
					 # Options: 2, 5, 10, 20 

input_shape = (320, 480)   # Input resolution. 
					 # Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

version = "combined" # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
					 # Options: "init", "combined"

# Camera options: baseline (m), focal length (pixel) and max distance
camera_config = CameraConfig(0.546, 500*input_shape[1]/1720) # rough estimate from the original calibration
scale_x = input_shape[1]/1762
scale_y = input_shape[0]/800
K = o3d.camera.PinholeCameraIntrinsic(width=input_shape[1], 
									  height=input_shape[0], 
									  fx=2014*scale_x, 
									  fy=2014*scale_y,
									  cx=906*scale_x, 
									  cy=398*scale_y)
max_distance = 20

# Initialize model
model_path = f'models/crestereo_{version}_iter{iters}_{input_shape[0]}x{input_shape[1]}.onnx'
depth_estimator = CREStereo(model_path, camera_config=camera_config, max_dist=max_distance)

# Get the driving stereo samples
driving_stereo_path = "drivingStereo"
start_sample = 700
left_images, right_images, depth_images = get_driving_stereo_images(driving_stereo_path, start_sample)
# out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1920, 1061))

# Initialize the Open3d visualizer
open3dVisualizer = Open3dVisualizer(K)

for left_path, right_path, depth_path in zip(left_images, right_images, depth_images):

	# Read frame from the video
	left_img = cv2.imread(left_path)
	right_img = cv2.imread(right_path)
	depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/1000

	# Estimate the depth
	depth_estimator(left_img, right_img)

	# Update 3D visualization
	open3dVisualizer(left_img, depth_estimator.depth_map*1000)	

	o3d_screenshot_mat = open3dVisualizer.vis.capture_screen_float_buffer()
	o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
	o3d_screenshot_mat = cv2.cvtColor(o3d_screenshot_mat, cv2.COLOR_RGB2BGR)
# 	out.write(o3d_screenshot_mat)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

# out.release()

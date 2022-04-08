import cv2
import numpy as np
import glob

from crestereo import CREStereo, CameraConfig

def get_driving_stereo_images(base_path, start_sample=0):

	# Get image list
	left_images = glob.glob(f'{base_path}/left/*.jpg')
	left_images.sort()
	right_images = glob.glob(f'{base_path}/right/*.jpg')
	right_images.sort()
	depth_images = glob.glob(f'{base_path}/depth/*.png')
	depth_images.sort()

	return left_images[start_sample:], right_images[start_sample:], depth_images[start_sample:]

# Store baseline (m) and focal length (pixel)
input_width = 640
camera_config = CameraConfig(0.546, 500*input_width/1720) # rough estimate from the original calibration
max_distance = 20

# Initialize model (Skip depth refinement for speed)
model_path = 'models/crestereo_sim.onnx'
model_half_path = 'models/crestereo_without_flow_sim.onnx'
depth_estimator = CREStereo(model_path, model_half_path, camera_config=camera_config, max_dist=max_distance)

# Get the driving stereo samples
driving_stereo_path = "drivingStereo"
start_sample = 700
left_images, right_images, depth_images = get_driving_stereo_images(driving_stereo_path, start_sample)
# out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (881,400*2))

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
for left_path, right_path, depth_path in zip(left_images, right_images, depth_images):

	# Read frame from the video
	left_img = cv2.imread(left_path)
	right_img = cv2.imread(right_path)
	depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/1000

	# Estimate the depth
	disparity_map = depth_estimator(left_img, right_img)
	color_depth = depth_estimator.draw_depth()

	# color_real_depth = depth_estimator.util_draw_depth(depth_img, (left_img.shape[1], left_img.shape[0]), max_distance)
	# combined_image = np.hstack((left_img, color_real_depth, color_depth))
	combined_image = np.vstack((color_depth,left_img))

	# out.write(combined_image)
	cv2.imshow("Estimated depth", combined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

# out.release()
cv2.destroyAllWindows()
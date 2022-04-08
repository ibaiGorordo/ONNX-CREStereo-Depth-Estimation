import cv2
import numpy as np
from imread_from_url import imread_from_url

from acvnet import ACVNet

resolutions = [(240,320),(320,480),(384,640),(480,640),(544,960),(720,1280)]

# Load images
left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

num_repetitions = 10 

for resolution in resolutions:

	print(f"Model: acvnet_maxdisp192_sceneflow_{resolution[0]}x{resolution[1]}.onnx")

	try:
		# Initialize model
		model_path = f'models/acvnet_maxdisp192_sceneflow_{resolution[0]}x{resolution[1]}/acvnet_maxdisp192_sceneflow_{resolution[0]}x{resolution[1]}.onnx'
		depth_estimator = ACVNet(model_path)

		
		for repetition in range(num_repetitions):

			# Estimate the depth
			disparity_map = depth_estimator(left_img, right_img)

		del depth_estimator
	except:
		print("Model could not be loaded")



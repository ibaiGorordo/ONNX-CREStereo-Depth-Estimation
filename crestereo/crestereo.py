import sys
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime

from performance_monitor import *

@dataclass
class CameraConfig:
    baseline: float
    f: float

DEFAULT_CONFIG = CameraConfig(0.546, 120) # rough estimate from the original calibration

class CREStereo():

	def __init__(self, model_path, model_half_path=None, camera_config=DEFAULT_CONFIG, max_dist=10):

		self.initialize_model(model_path, model_half_path, camera_config, max_dist)

	def __call__(self, left_img, right_img):

		return self.update(left_img, right_img)

	def initialize_model(self, model_path, model_half_path=None, camera_config=DEFAULT_CONFIG, max_dist=10):

		self.camera_config = camera_config
		self.max_dist = max_dist
		self.session_half = None

		# Initialize model session
		self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider',
																		   'CPUExecutionProvider'])
		
		# Get model info
		self.get_input_details()
		self.get_output_details()

		# Add the half resolution model if passed
		if model_half_path:
			self.session_half = onnxruntime.InferenceSession(model_half_path, providers=['CUDAExecutionProvider',
																			   			 'CPUExecutionProvider'])
		elif len(self.input_names) > 2:
			print("You need to pass either a model without flow_init input, or an additional model to estimate flow_init")
			sys.exit(0)

	def update(self, left_img, right_img):

		flow_init = None

		if self.session_half:

			left_tensor_half = self.prepare_input(left_img, half=True)
			right_tensor_half = self.prepare_input(right_img, half=True)

			flow_init = self.inference(left_tensor_half, right_tensor_half)

		left_tensor = self.prepare_input(left_img)
		right_tensor = self.prepare_input(right_img)

		# Estimate the disparity map
		outputs = self.inference(left_tensor, right_tensor, flow_init)
		self.disparity_map = self.process_output(outputs)

		# Estimate depth map from the disparity
		self.depth_map = self.get_depth_from_disparity(self.disparity_map, self.camera_config)

		return self.disparity_map

	def prepare_input(self, img, half=False):

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		self.img_height, self.img_width = img.shape[:2]

		if half:
			img_input = cv2.resize(img, (self.input_width//2,self.input_height//2))
		else:
			img_input = cv2.resize(img, (self.input_width, self.input_height))

		img_input = img_input.transpose(2, 0, 1)
		img_input = img_input[np.newaxis,:,:,:]        

		return img_input.astype(np.float32)

	@performance
	def inference(self, left_tensor, right_tensor, flow_init=None):

		if self.session_half and flow_init is None:

			return self.session_half.run(self.output_names, {self.input_names[0]: left_tensor,
													         self.input_names[1]: right_tensor})[0]
		elif flow_init is None:

			return self.session.run(self.output_names, {self.input_names[0]: left_tensor,
										                self.input_names[1]: right_tensor})[0]

		else:
			return self.session.run(self.output_names, {self.input_names[0]: left_tensor,
										                self.input_names[1]: right_tensor,
										                self.input_names[2]: flow_init})[0]

	def process_output(self, output): 

		return np.squeeze(output[:,0,:,:])

	@staticmethod
	def get_depth_from_disparity(disparity_map, camera_config):

		return camera_config.f*camera_config.baseline/disparity_map

	def draw_disparity(self):

		disparity_map =  cv2.resize(self.disparity_map,  (self.img_width, self.img_height))
		norm_disparity_map = 255*((disparity_map-np.min(disparity_map))/
								  (np.max(disparity_map)-np.min(disparity_map)))

		return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)

	def draw_depth(self):
		
		return self.util_draw_depth(self.depth_map, (self.img_width, self.img_height), self.max_dist)

	@staticmethod
	def util_draw_depth(depth_map, img_shape, max_dist):

		norm_depth_map = 255*(1-depth_map/max_dist)
		norm_depth_map[norm_depth_map < 0] = 0
		norm_depth_map[norm_depth_map >= 255] = 0

		norm_depth_map =  cv2.resize(norm_depth_map, img_shape)

		return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

		self.output_shape = model_outputs[0].shape

if __name__ == '__main__':
	
	from imread_from_url import imread_from_url

	# Initialize model
	model_path = '../models/crestereo_sim.onnx'
	model_half_path = '../models/crestereo_without_flow_sim.onnx'
	depth_estimator = CREStereo(model_path, model_half_path)

	# Load images
	left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
	right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

	# Estimate depth and colorize it
	disparity_map = depth_estimator(left_img, right_img)
	color_disparity = depth_estimator.draw_disparity()

	combined_img = np.hstack((left_img, color_disparity))

	cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
	cv2.imshow("Estimated disparity", combined_img)
	cv2.waitKey(0)

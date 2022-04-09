import os
import time
import cv2
import numpy as np
import glob
import pandas as pd
import re

import sys
sys.path.insert(1, '../')

from crestereo import CREStereo, CameraConfig

def get_driving_stereo_images(base_path="../drivingStereo", start_sample=0):

	# Get image list
	left_images = glob.glob(f'{base_path}/left/*.png')
	left_images.sort()
	right_images = glob.glob(f'{base_path}/right/*.png')
	right_images.sort()
	depth_images = glob.glob(f'{base_path}/depth/*.png')
	depth_images.sort()

	return left_images[start_sample:], right_images[start_sample:], depth_images[start_sample:]

def get_best_possible_results():

	# Uses the largest model to get the best possible disparity maps for comparison
	best_model_path = '../models/crestereo_combined_iter20_720x1280.onnx'
	best_depth_estimator = CREStereo(best_model_path)

	best_disparities = []
	for img_id in test_img_ids:

		left_img = cv2.imread(left_images[img_id])
		right_img = cv2.imread(right_images[img_id])
		disparity_map = best_depth_estimator(left_img, right_img)
		best_disparities.append(disparity_map)

	return best_disparities

def calculate_disp_diff(disp_map, ref_disp_map):

	# Resize disparity to match reference map
	ref_heigth, ref_width = ref_disp_map.shape[:2]
	disp_map =  cv2.resize(disp_map, (ref_width, ref_heigth), cv2.INTER_CUBIC)

	return np.abs(ref_disp_map - disp_map)

if __name__ == '__main__':
	
	# List images
	left_images, right_images, _ = get_driving_stereo_images()

	# List models
	models = glob.glob('../models/*.onnx')

	# Test parameters:
	num_images = 50

	# Select the test images randomly
	rng = np.random.default_rng()
	test_img_ids = rng.choice(len(left_images), size=num_images, replace=False)

	# Get the best possible disparities with the largest model
	best_disparities = get_best_possible_results()

	# Compare the rest of the results with the largest model
	results_df = pd.DataFrame(columns =['model','type','iters','height','width','avg-disp-diff','avg-inf-time'])
	for model_num, model_path in enumerate(models):

		model_name = os.path.basename(model_path)
		model_params = re.findall('crestereo_([a-z]+)_iter([0-9]+)_([0-9]+)x([0-9]+)[.]onnx$', model_name)[0]
		if "next" in model_params:
			continue

		depth_estimator = CREStereo(model_path)

		print(f"Testing {model_name} model")

		model_inference_times = []
		model_diffs = []
		for img_num, img_id in enumerate(test_img_ids):

			left_img = cv2.imread(left_images[img_id])
			right_img = cv2.imread(right_images[img_id])

			disparity_map = depth_estimator(left_img, right_img)
			
			epe = calculate_disp_diff(disparity_map, best_disparities[img_num])
			model_diffs.append(epe)
			model_inference_times.append(depth_estimator.inf_time)

		# Skip the first inference as it has larger duration than normal
		model_inference_times.pop(0)

		avg_inf_time = np.array(model_inference_times).mean()
		avg_diff = np.array(model_diffs).mean()


		model_df = pd.DataFrame({'model': model_name,
								 'type': model_params[0],
								 'iters': model_params[1],
								 'height': model_params[2],
								 'width': model_params[3],
								 'avg-disp-diff': avg_diff, 
								 'avg-inf-time': avg_inf_time}, index=[model_num])
		results_df = pd.concat([results_df,model_df])
		print(f"Avg Inf. time: {int(avg_inf_time*1000)} ms, Avg. Disp. Diff: {avg_diff:.3f} px\n")

	# Save results
	results_df.to_csv("model_option_comparison.csv",index=False)

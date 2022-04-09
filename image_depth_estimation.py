import cv2
import numpy as np
from imread_from_url import imread_from_url

from crestereo import CREStereo

# Model Selection options (not all options supported together)
iters = 5            # Lower iterations are faster, but will lower detail. 
		             # Options: 2, 5, 10, 20 

shape = (240, 320)   # Input resolution. 
				     # Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

version = "combined" # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
					 # Options: "init", "combined"

# Initialize model
model_path = f'models/crestereo_{version}_iter{iters}_{shape[0]}x{shape[1]}.onnx'
depth_estimator = CREStereo(model_path)

# Load images
left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

# Estimate the depth
disparity_map = depth_estimator(left_img, right_img)

color_disparity = depth_estimator.draw_disparity()
combined_image = np.hstack((left_img, color_disparity))

cv2.imwrite("out.jpg", combined_image)

cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
cv2.imshow("Estimated disparity", combined_image)
cv2.waitKey(0)

cv2.destroyAllWindows()

import numpy as np
import cv2
import open3d as o3d

class Open3dVisualizer():

	def __init__(self, K):

		self.point_cloud = o3d.geometry.PointCloud()
		self.o3d_started = False
		self.K = K

		self.vis = o3d.visualization.Visualizer()
		self.vis.create_window()
	
	def __call__(self, rgb_image, depth_map, max_dist=20):

		self.update(rgb_image, depth_map, max_dist)

	def update(self, rgb_image, depth_map, max_dist=20):

		# Prepare the rgb image
		rgb_image_resize = cv2.resize(rgb_image, (depth_map.shape[1],depth_map.shape[0]))
		rgb_image_resize = cv2.cvtColor(rgb_image_resize, cv2.COLOR_BGR2RGB)
		rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_image_resize), 
									                               o3d.geometry.Image(depth_map),
									                               1, depth_trunc=max_dist*1000, 
									                               convert_rgb_to_intensity = False)
		temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.K)
		temp_pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

		# Add values to vectors
		self.point_cloud.points = temp_pcd.points
		self.point_cloud.colors = temp_pcd.colors

		# Add geometries if it is the first time
		if not self.o3d_started:
			self.vis.add_geometry(self.point_cloud)
			self.o3d_started = True

			# Set camera view
			ctr = self.vis.get_view_control()
			ctr.set_front(np.array([ -0.0053112027751292369, 0.28799919460714768, 0.95761592250270977 ]))
			ctr.set_lookat(np.array([-78.783105080589237, -1856.8182240774879, -10539.634663481682]))
			ctr.set_up(np.array([-0.029561736688513099, 0.95716567219818627, -0.28802774118017438]))
			ctr.set_zoom(0.31999999999999978)

		else:
			self.vis.update_geometry(self.point_cloud)

		self.vis.poll_events()
		self.vis.update_renderer()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import pandas as pd

results_df = pd.read_csv ('model_option_comparison.csv')
results_df["avg-inf-time"] *= 1000 # ms

# Generete input size colors
unique_sizes = np.sort((results_df['height']*results_df['width']).unique())
size_dict = {size:i/(len(unique_sizes)-1) for i,size in enumerate(unique_sizes)}
input_shape_cm = cm.get_cmap('tab10', len(unique_sizes))

# Split results for init and combined and size
results_init_120x160 = results_df[(results_df["type"] == "init") &
								  (results_df["height"] == 120 ) &
	                              (results_df["width"] == 160) ]
results_init_160x240 = results_df[(results_df["type"] == "init") &
								  (results_df["height"] == 160 ) &
	                              (results_df["width"] == 240) ]
results_init_180x320 = results_df[(results_df["type"] == "init") &
								  (results_df["height"] == 180 ) &
	                              (results_df["width"] == 320) ]
results_init_240x320 = results_df[(results_df["type"] == "init") &
								  (results_df["height"] == 240 ) &
	                              (results_df["width"] == 320) ]
results_init_360x640 = results_df[(results_df["type"] == "init") &
								  (results_df["height"] == 360 ) &
	                              (results_df["width"] == 640) ]
results_init_480x640 = results_df[(results_df["type"] == "init") &
								  (results_df["height"] == 480 ) &
	                              (results_df["width"] == 640) ]	                              
results_init_1280x720 = results_df[(results_df["type"] == "init") &
								  (results_df["height"] == 720 ) &
	                              (results_df["width"] == 1280) ]
results_comb_240x320 = results_df[(results_df["type"] == "combined") &
								  (results_df["height"] == 240 ) &
	                              (results_df["width"] == 320) ]
results_comb_320x480 = results_df[(results_df["type"] == "combined") &
								  (results_df["height"] == 320 ) &
	                              (results_df["width"] == 480) ]
results_comb_360x640 = results_df[(results_df["type"] == "combined") &
								  (results_df["height"] == 360 ) &
	                              (results_df["width"] == 640) ]
results_comb_480x640 = results_df[(results_df["type"] == "combined") &
								  (results_df["height"] == 480 ) &
	                              (results_df["width"] == 640) ]	                              
results_comb_1280x720 = results_df[(results_df["type"] == "combined") &
								  (results_df["height"] == 720 ) &
	                              (results_df["width"] == 1280) ]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(results_init_120x160["avg-inf-time"], 
	        results_init_120x160["avg-disp-diff"], 
	        label="init_120x160",
	        s=results_init_120x160["iters"]**2, marker='o',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_init_120x160["width"]*results_init_120x160["height"])))

ax.scatter(results_init_160x240["avg-inf-time"], 
	        results_init_160x240["avg-disp-diff"], 
	        label="init_160x240",
	        s=results_init_160x240["iters"]**2, marker='o',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_init_160x240["width"]*results_init_160x240["height"])))

ax.scatter(results_init_180x320["avg-inf-time"], 
	        results_init_180x320["avg-disp-diff"], 
	        label="init_180x320",
	        s=results_init_180x320["iters"]**2, marker='o',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_init_180x320["width"]*results_init_180x320["height"])))

ax.scatter(results_init_240x320["avg-inf-time"], 
	        results_init_240x320["avg-disp-diff"], 
	        label="init_240x320",
	        s=results_init_240x320["iters"]**2, marker='o',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_init_240x320["width"]*results_init_240x320["height"])))

ax.scatter(results_init_360x640["avg-inf-time"], 
	        results_init_360x640["avg-disp-diff"], 
	        label="init_360x640",
	        s=results_init_360x640["iters"]**2, marker='o',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_init_360x640["width"]*results_init_360x640["height"])))

ax.scatter(results_init_480x640["avg-inf-time"], 
	        results_init_480x640["avg-disp-diff"], 
	        label="init_480x640",
	        s=results_init_480x640["iters"]**2, marker='o',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_init_480x640["width"]*results_init_480x640["height"])))

ax.scatter(results_comb_240x320["avg-inf-time"], 
	        results_comb_240x320["avg-disp-diff"], 
	        label="comb_240x320",
	        s=results_comb_240x320["iters"]**2, marker='^',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_comb_240x320["width"]*results_comb_240x320["height"])))

ax.scatter(results_comb_320x480["avg-inf-time"], 
	        results_comb_320x480["avg-disp-diff"], 
	        label="comb_320x480",
	        s=results_comb_320x480["iters"]**2, marker='^',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_comb_320x480["width"]*results_comb_320x480["height"])))

ax.scatter(results_comb_360x640["avg-inf-time"], 
	        results_comb_360x640["avg-disp-diff"], 
	        label="comb_360x640",
	        s=results_comb_360x640["iters"]**2, marker='^',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_comb_360x640["width"]*results_comb_360x640["height"])))

ax.scatter(results_comb_480x640["avg-inf-time"], 
	        results_comb_480x640["avg-disp-diff"], 
	        label="comb_480x6400",
	        s=results_comb_480x640["iters"]**2, marker='^',  alpha=.5,
	        c=input_shape_cm(np.vectorize(size_dict.get)(results_comb_480x640["width"]*results_comb_480x640["height"])))

# Major ticks every 20, minor ticks every 5
x_ticks = np.arange(0, 1000, 100)
y_ticks = np.arange(18, 35, 1)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

ax.grid(which='major', alpha=0.3)
plt.xlabel('Inference time (ms)')
plt.ylabel('Avg. disparity difference (px)')
plt.title('CREStereo model options comparison')
plt.tight_layout()
plt.legend(loc='best')
plt.show()

# ONNX-CREStereo-Depth-Estimation
 Python scripts performing stereo depth estimation using the CREStereo model in ONNX.
 
![!CREStereo detph estimation](https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation/blob/main/doc/img/out.jpg)
*Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)*

# Requirements

 * Check the **requirements.txt** file. 
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.
 * For OAK-D host inference, you will need the **depthai** library.
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation.git
cd ONNX-CREStereo-Depth-Estimation
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

### For youtube video inference
```
pip install youtube_dl
pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

### OAK-D Host inference:
```pip install depthai```

You might need additional installations, check the depthai reference below for more details.

# ONNX model
The models were converted from the Pytorch implementation below by [PINTO0309](https://github.com/PINTO0309), download the models from the download script in [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/284_CREStereo) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation/tree/main/models)** folder. 
- The License of the models is Apache-2.0 License: https://github.com/megvii-research/CREStereo/blob/master/LICENSE

# Original MegEngine model
The original model was trained in the MegEngine framework: [original repository](https://github.com/megvii-research/CREStereo).

# Pytorch model
The original MegEngine model was converted to Pytorch with this repository: https://github.com/ibaiGorordo/CREStereo-Pytorch
 
# Examples

 * **Image inference**:
 ```
 python image_depth_estimation.py
 ```

 * **Video inference**:
 ```
 python video_depth_estimation.py
 ```
 
 * **Driving Stereo dataset inference**: https://youtu.be/ciX7ILgpJtw
 ```
 python driving_stereo_test.py
 ```
 ![!CREStereo depth estimation](https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation/blob/main/doc/img/crestereo.gif)
  
 *Original video: Driving stereo dataset, reference below*

 * **Driving Stereo 3D point cloud visualization**: https://youtu.be/vlBbH28PgHk
 ```
 python driving_stereo_point_cloud.py
 ```
 ![!CREStereo depth estimation point cloud](https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation/blob/main/doc/img/crestereo_point.gif)
  
 *Original video: Driving stereo dataset, reference below*
  
  

 * **Depthai inference**: 
 ```
 python depthai_host_depth_estimation.py
 ```
# Model option comparison (Nvidia 1660 Super)
In the graph below, the different model options, i.e. input shape, version (init or combined) and number of iterations are combined. The comparison is done compared to the results obtained with the largest model (720x1280 combined with 20 iters), as it is expected to provide the best results. 
- The size of the marker indicates the number of iterations, increasing with the number of iterations.
![!CREStereo model option comparison](https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation/blob/main/doc/img/crestereo_options_comp.png)

# References:
* CREStereo model: https://github.com/megvii-research/CREStereo
* CREStereo - Pytorch: https://github.com/ibaiGorordo/CREStereo-Pytorch
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* Driving Stereo dataset: https://drivingstereo-dataset.github.io/
* Depthai library: https://pypi.org/project/depthai/
* Original paper: https://arxiv.org/abs/2203.11483

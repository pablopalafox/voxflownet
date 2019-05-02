# VoxFlowNet
Learning Scene Flow in Point Clouds through Voxel Grids

<!-- | | |
|:-------------------------:|:-------------------------:|
|<img alt="test_3" src="/assets/images/test_munich/test_3.png">  |  <img alt="test_3_output" src="/assets/images/test_munich/test_3_output.png">|
|<img alt="test_3_ALL" src="/assets/images/test_munich/test_3_ALL.png">  |  <img alt="test_3_planes" src="/assets/images/test_munich/test_3_planes.png">|

<p align="center">
	<img src="/assets/images/result_stuttgart.gif" alt="result_on_stuttgart_video">
</p>

<a name="intro"></a> -->


<p align="center">
	<img src="https://github.com/pablorpalafox/voxflownet/blob/master/doc/pipeline.png" alt="pipeline">
</p>



This work was done as part of my Guided Research at the [Visual Computing Lab at TUM](https://www.niessnerlab.org/) under the supervision of [Prof. Matthias Niessner](https://www.niessnerlab.org/members/matthias_niessner/profile.html). For more info on the project, check my [report](/assets/report.pdf).

Author: [Pablo Rodriguez Palafox](https://pablorpalafox.github.io/)  
Supervisor: [Prof. Matthias Niessner](https://www.niessnerlab.org/members/matthias_niessner/profile.html)  
[Visual Computing Group at TUM](https://www.niessnerlab.org/)  
[Technical University Munich](https://www.tum.de/)  



## 1. Requirements (& Installation tips)
This code was tested with Pytorch 1.0.0, CUDA 10.0 and Ubuntu 16.04.

You can set your own python environment and install the required dependencies using the [environment.yaml](environment.yaml).



## 2. Dataset - FlyingThings3D

The data preprocessing tools are in `generate_dataset`. Firts download the raw [FlyingThings3D dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). We need `flyingthings3d__disparity.tar.bz2`, `flyingthings3d__disparity_change.tar.bz2`, `flyingthings3d__optical_flow.tar.bz2` and `flyingthings3d__frames_finalpass.tar`. Then extract the files in `/path/to/flyingthings3d` and make sure that the directory looks like so:

```
/path/to/flyingthings3d
  disparity/
  disparity_change/
  optical_flow/
  frames_finalpass/
```

Then 'cd' into 'generate_dataset' and execute the following command:

```bash
python generate_Flying.py
```

## 3. Usage

### Training

Make sure that the 'do_train' flag is set to 'True' in 'config.py'. Also configure the number of 'epochs', 'batch_sizes' and other stuff to what you need. By setting 'OVERFIT' to 'True' you can just overfit to some examples, which can be set in 'sequences_to_train'.

```bash
python main.py
```

### Evaluation

In 'config.py' set the 'do_train' flag to 'False'. Set 'model_dir_to_use_at_eval' to the name of the model you want to evaluate.

```bash
python main.py
```


## 4. License

Our code is released under MIT License (see LICENSE file for details).

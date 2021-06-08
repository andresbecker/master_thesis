<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">

<h3 align="center">Predicting transcription rate from multiplexed protein maps using deep learning</h3>

  <a href="https://github.com/andresbecker/master_thesis">
    <p align="center">
      <img src="workspace/Interpretability/Gradient/Gradient.gif" width="600">
      <img src="workspace/Interpretability/IG_plots/VG.gif" width="600">
    </p>
  </a>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

<p align="center">
  <img src="Thesis_defense_presentation/work_flow.JPG" width="800">
</p>

By means of fluorescent antibodies it is possible to observe the amount of nascent RNA within the nucleus of a cell, and thus estimate its `Transcription Rate (TR)`. But what about the other molecules, proteins, organelles, etc. within the nucleus of the cell? Is it possible to estimate the  using only the shape and distribution of these subnuclear components? By means of multichannel images of single cell nucleus (obtained through the `Multiplexed Protein Maps (MPM)` protocol (cite Guteaar7042)) and `Convolutional Neural Networks (CNNs)`, we show that this is possible.
Applying pre-processing and data augmentation techniques, we reduce the information contained in the intensity of the pixels and the correlation of these between the different channels. This allowed the CNN to focus mainly on the information provided by the location, size and distribution of elements within the cell nucleus.
For this task different architectures were tried, from a simple CNN (with only 160k parameters), to more complex architectures such as the ResNet50V2 or the Xception (with more than 20m parameters).
Furthermore, through the interpretability methods `Integrated Gradients (IG)` (cite here) and `VarGrad (VG)` (cite here), we could obtain score maps that allowed us to observe the pixels that the CNN considered as relevant to predict the TR for each cell nucleus input image. The analysis of these score maps reveals how as the TR changes, the CNN focuses on different proteins and areas of the nucleus. This shows that interpretability methods can help us to understand how a CNN make its predictions and learn from it, which has the potential to provide guidance for new discoveries in the field of biology.

You can find the complete explanation and development of this work in <a href="https://github.com/andresbecker/master_thesis/blob/main/Manuscript/Thesis_Andres_Becker.pdf"><strong>`Manuscript/Thesis_Andres_Becker.pdf`»</strong></a>

### Built With

* [Anaconda 4.10.1](https://www.anaconda.com/)
* [Python 3.8](https://www.python.org/downloads/release/python-380/)
* [TensorFlow 2.5](https://www.tensorflow.org/tutorials/quickstart/beginner)

**Important** <br>
There is a bug in the TensorFlow function `tf.image.central_crop` that does not allow to take a tensor as input for the argument `central_fraction`, which is needed for this work. This bug was fixed since the TensorFlow version 2.5. Therefore, you can either use TF 2.5 or replace manually the library `image_ops_impl.py` in your local machine by [this](https://raw.githubusercontent.com/tensorflow/tensorflow/b7a7f8d178254d1361d34dfc40a58b8dce48b9d7/tensorflow/python/ops/image_ops_impl.py). <br>
Reference: https://github.com/tensorflow/tensorflow/pull/45613/files.


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

A running installation of Anaconda. If you haven't installed Anaconda yet, you can follow the next tutorial: <br>
[Anaconda Installation](https://docs.anaconda.com/anaconda/install/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/andresbecker/master_thesis.git
   ```
2. You can install the environment either by loading the `YML` file
   ```sh
   conda env create -f conda_environment.yml
   ```
   or step by step

   a. Create and activate the environment
      ```sh
      conda create -n mpm_inter_env python=3.8
      conda activate mpm_inter_env
      ```
   b. Install the needed packages
      ```sh
      conda install tensorflow=2.5 tensorboard tensorflow-datasets numpy
      conda install matplotlib seaborn
      conda install jupyterlab
      # To build TensorFlow Datasets
      pip install -q tfds-nightly
      ```

<!-- USAGE EXAMPLES -->
## Usage

This implementation is divided in 4 main steps

0. Activate the environment and open jupyter-lab
```sh
conda activate mpm_inter_env
jupyter-lab
```

1. Raw data preprocessing (transformation from text files to multichannel images of single cell nucleus).
    ```sh
    conda activate mpm_inter_env
    jupyter-lab
    ```
    Open the notebook [`workspace/notebooks/MPPData_into_images_no_split.ipynb`](https://github.com/andresbecker/master_thesis/blob/main/workspace/notebooks/MPPData_into_images_no_split.ipynb)  using the Jupyter navigator and replace the variable `PARAMETERS_FILE` with the absolute path and name of the file containing your input parameters. You can find the parameter file used for this work [here](workspace/scripts/Data_Preprocessing/Parameters/MppData_to_imgs_no_split.json).<br>
    Also, you can look at a dummy example (and parameters) of the raw data preprocessing in [this notebook](https://github.com/andresbecker/master_thesis/blob/main/workspace/notebooks/MPPData_into_images_no_split_dummy.ipynb). <br>
    You can find an explanation of the preprocessing input parameters on appendix `A.1` of [`Manuscript/Thesis_Andres_Becker.pdf`](https://github.com/andresbecker/master_thesis/blob/main/Manuscript/Thesis_Andres_Becker.pdf).

2. TensorFlow dataset (TFDS) creation.
    a. Go to the directory where the python scripts to create the TFDSs are
      ```sh
      cd workspace/tf_datasets_scripts
      ```
    b. Specify the parameters for the dataset (like perturbations, wells, output channel, etc)
      ```sh
      vi ./MPP_DS_normal_DMSO_z_score/Parameters/my_tf_dataset_parameters.json
      ```
    c. Build the TFDS using the script [`Create_tf_dataset.sh`](https://github.com/andresbecker/master_thesis/blob/main/workspace/tf_datasets_scripts/Create_tf_dataset.sh)
      ```sh
      ./Create_tf_dataset.sh -o /path_to_store_the_TFDS/tensorflow_datasets -n MPP_DS_normal_DMSO_z_score -p ./MPP_DS_normal_DMSO_z_score/Parameters/my_tf_dataset_parameters.json -e mpm_inter_env
      ```

  You can find the parameter file used for this work [here](workspace/tf_datasets_scripts/MPP_DS_normal_DMSO_z_score/Parameters/tf_dataset_parameters_server.json).<br>
  Also, you can build a dummy TFDS (and parameters) by executing the following <br>
  ```sh
  cd /path_to_this_repo/workspace/tf_datasets_scripts
  ./Create_tf_dataset.sh -o /data/Master_Thesis_data/datasets/tensorflow_datasets -n MPP_DS_normal_DMSO_z_score_dummy -p ./MPP_DS_normal_DMSO_z_score_dummy/Parameters/tf_dataset_parameters_dummy.json -e mpm_inter_env
  ```
  You can find an explanation of the input parameters to build a TFDS on appendix `A.2` of [`Manuscript/Thesis_Andres_Becker.pdf`](https://github.com/andresbecker/master_thesis/blob/main/Manuscript/Thesis_Andres_Becker.pdf).

3. Model training <br>
Open the notebook `workspace/notebooks/Model_training_class.ipynb` using the Jupyter navigator and replace the variable `PARAMETERS_FILE` with the absolute path and name of the file containing your input parameters. You can find the parameters files used in this work [here](workspace/scripts/Model_training/Thesis_final_results/Parameters).<br>
Also, you can look at a dummy model training example (and parameters) in [this notebook](https://github.com/andresbecker/master_thesis/blob/main/workspace/notebooks/Model_training_class_dummy.ipynb). <br>
You can find an explanation of the model training input parameters on appendix `A.3` of [`Manuscript/Thesis_Andres_Becker.pdf`](https://github.com/andresbecker/master_thesis/blob/main/Manuscript/Thesis_Andres_Becker.pdf).

4. Model interpretation. <br>
a. Go to the directory where the python scripts for interpretability methods are <br>
  ```sh
  cd workspace/Interpretability/Python_scripts
  ```
b. Specify the parameters for the interpretability methods (IG number of steps, output dir, etc.) <br>
  ```sh
  vi ./Parameters/my_parameters_file.json
  ```
c. Create the score maps using the python script [`get_VarGradIG_from_TFDS_V2.py`](https://github.com/andresbecker/master_thesis/blob/main/workspace/Interpretability/Python_scripts/get_VarGradIG_from_TFDS_V2.py) <br>
  ```sh
  python get_VarGradIG_from_TFDS_V2.py -i ./Parameters/my_parameters_file.json
  ```

You can find the parameter file used for this work [here](workspace/tf_datasets_scripts/MPP_DS_normal_DMSO_z_score/Parameters/tf_dataset_parameters_server.json).
Also, you can create dummy score maps using the Bash script [`Run_pyhton_script.sh`](https://github.com/andresbecker/master_thesis/blob/main/workspace/Interpretability/Python_scripts/Run_pyhton_script.sh) <br>
```sh
./Run_pyhton_script.sh -e mpm_inter_env -s ./get_VarGradIG_from_TFDS_V2.py -p ./Parameters/Simple_CNN_dummy.json
```
You can find an explanation of the input parameters for the interpretability methods on appendix `A.4` of [`Manuscript/Thesis_Andres_Becker.pdf`](https://github.com/andresbecker/master_thesis/blob/main/Manuscript/Thesis_Andres_Becker.pdf).

<!-- CONTACT -->
## Contact

Andres Becker - [LinkedIn](https://www.linkedin.com/in/andres-becker) - andres.becker@tum.de

Project Link: [https://github.com/andresbecker/master_thesis](https://github.com/andresbecker/master_thesis)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Hado Van Hasselt, Advanced Deep Learning and Reinforcement Learning. Lecture 12: Policy Gradients and Actor Critics](https://youtu.be/bRfUxQs6xIM)
* [Abhishek Suran, Actor-Critic with tf-2-x](https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-1-of-2-d1e26a54ce97)
* [Lilian Weng, A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)
* [Lilian Weng, Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)

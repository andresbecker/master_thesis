<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/andresbecker/master_thesis">
    <img src="workspace/Interpretability/Gradient/Gradient.gif" width="400">
  </a>

  <h3 align="center">Predicting transcription rate from multiplexed protein maps using deep learning</h3>

  <p align="center">
    Implementation of an Advantage Actor-Critic using Artificial Neural Networks
    <br />
    <a href="https://github.com/andresbecker/Deep_RL_Actor_Critic/blob/main/References/A2C_Summary/A2C_Summary.pdf"><strong>Explore the docs »</strong></a>
  </p>
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
  <img src="Resources/Imp_diagram.png" width="500">
</p>

This is a very simple implementation of a Deep Reinforcement Learning Advantage Actor-Critic. It uses 2 independent Artificial Neural Networks to approximate the Policy function (Actor) and the State-value function (Critic). To test the implementation, I use the Moon Lander environment provided by OpenAI-Gym.

If you want to have a deeper understanding of the Actor-Critic algorithm, I strongly recommend you to take a look into the document `References/A2C_Summary/A2C_Summary.pdf` and `References/A2C_Presentation.pdf`. In `References/A2C_Summary/` you can also find the original $\\LaTeX$ document used to create the summary.

### Built With

* [Anaconda 4.9](https://www.anaconda.com/)
* [TensorFlow 2.2](https://www.tensorflow.org/tutorials/quickstart/beginner)
* [OpenAI Gym](https://gym.openai.com/)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

A running installation of Anaconda. If you haven't installed Anaconda yet, you can follow the next tutorial: <br>
[Anaconda Installation](https://docs.anaconda.com/anaconda/install/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/andresbecker/Deep_RL_Actor_Critic.git
   ```
2. Install the environment
   ```sh
   conda env create -f conda_environment.yml
   ```

<!-- USAGE EXAMPLES -->
## Usage

To train and test this implementation, simply activate the environment
```sh
conda activate A2C_env
```
open jupyter-lab
```sh
jupyter-lab
```
and navigate to open the notebook `A2C.ipynb`.
Then, just follow the steps inside the notebook.

Have fun!

<!-- CONTACT -->
## Contact

Andres Becker - [LinkedIn](https://www.linkedin.com/in/andres-becker) - andres.becker@tum.de

Project Link: [https://github.com/andresbecker/Deep_RL_Actor_Critic](https://github.com/andresbecker/Deep_RL_Actor_Critic)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Hado Van Hasselt, Advanced Deep Learning and Reinforcement Learning. Lecture 12: Policy Gradients and Actor Critics](https://youtu.be/bRfUxQs6xIM)
* [Abhishek Suran, Actor-Critic with tf-2-x](https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-1-of-2-d1e26a54ce97)
* [Lilian Weng, A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)
* [Lilian Weng, Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)

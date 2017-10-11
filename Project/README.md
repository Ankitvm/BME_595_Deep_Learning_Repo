# Title: 

# **Indoor Place Categorization for Visual SLAM**

## Team members

### Ankit Manerikar 

## Goals

The problem of visual place categorization (VPC), i.e., the visual identification of an input scene as belonging to a particular semantic pre-trained category is a well-established one and has been tackled through the development of a large number of datasets as well as the design of CNNs using these datasets that give a high value of accuracy. For example, the Places205-AlexNet [1] trained on the MIT Places Scene Recognition Database [2] allows for a test accuracy of more than 95 % for the case of scene recognition. However, extending this CNN-based place categorization to Visual SLAM systems can result in a few implementational constraints as follows:

- While place recognition for vision systems operates on 2D instantaneous image inputs, in case of visual SLAM, place recognition needs to be carried out on 3D maps or sections of maps that are built during the propagation of the SLAM robot.
- As a result place categorization in SLAM requires a spatial/temporal consistency in the output that may not always be guarranteed by the VPC classifier - this is because VPC basically operates on the spatio-temporal data in the 3D physical place corresponding to the contsructed visual map thus basically parititioning the map into different semantic place labels.
- Now, there exist datasets and classifiers to train CNNs for categorization of SLAM maps but very few implementation have the implementational robustness that can otherwise be obtained from visual place categorization.

Hence, this project aims at developing a two-tier technique that provides an extension of CNNs trained for classification of 2D Visual Place Categorization to that of paritioning maps constructed from SLAM using semantic place labels.

## Challenges

Visual SLAM systems operate upon an input stream of images (either monocular, stereo or RGB-D) to construct a 3D Map of the environment while performing localization of the robot simultaneously. Since the sensory input to such a system is an image, it is only convenient to use CNNs trained for VPC to classify these input image frames using pre-trained labels as they are streamed for the purpose of SLAM. However, categorization of the constructed maps in a similar manner would require these labels that are instantaneously tagged to the input image frames be extended to classify section of maps as  well. The main challenge is therefore to utilize these spatio-temporally generated images labels as feature vectors for parititioning the entire map into different places. 

This can be done using a two-tier technique [3] employing two classifiers back-to-back as follows:

- The first classifier is a VPC CNN trained on a Place Recognition dataset that operates on the 2D input images and assigns place labels to current pose of the robot, thus generating a spatio-temporal grid of class labels corresponding to the path traversed by the robot.
- The second classifier takes as input this grid of place labels and performs a semantic segmentation of the grid to partition the same into different places.

Implementation of these two classifiers is the main challenge for the successful execution of the project.



## References

[1] Zhou, Bolei, et al. "Learning deep features for scene recognition using places database." Advances in neural 	information processing systems. 2014.

[2] Wang, Limin, et al. "Places205-vggnet models for scene recognition." arXiv preprint arXiv:1508.01667 (2015).

[3] Sünderhauf, Niko, et al. "Place categorization and semantic mapping on a mobile robot." Robotics and Automation (ICRA), 2016 IEEE International Conference on. IEEE, 2016.

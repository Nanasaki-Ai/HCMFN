# HCMFN
**Human-Centric Multimodal Fusion Network for Robust Action Recognition**

# Abstract
The core idea of the proposed method is to address the challenge of effectively fusing different modalities in human action recognition. To overcome the presence of redundant background information, the method converts RGB, depth, and optical flow data into human-centric images (HCI) based on key point sequences. HCI prioritize the movement changes of essential body parts, minimizing the influence of unnecessary backgrounds and reducing computing resource requirements. Additionally, a human-centric multimodal fusion network (HCMFN) is introduced to learn action features from different modalities and extract comprehensive action patterns. 

# Acknowledgements

 This work is based on the following two works：

 **MMNet**, TPAMI 2022, [Original code](https://github.com/bruceyo/MMNet)
 
 **PoseC3D**, CVPR 2022, [Original code](https://github.com/kennymckormick/pyskl)

 Thanks to the original authors for their work! Although our work represents only a modest improvement upon existing studies, we remain optimistic that it can provide valuable enlightenment to someone.

 Meanwhile, we are very grateful to the creators of these two datasets, i.e., NTU RGB+D and NTU RGB+D 120. Your selfless work has made a great contribution to the computer vision community!

 Last but not least, the authors will be very grateful for the selfless and constructive suggestions of the reviewers.
 
  # How to reproduce this work?
 To replicate this work, you must complete several stages, encompassing [dataset download and preprocess](#download-dataset-and-preprocess), [establishment of the operating environment](#build-environment), [model train and test](#train-and-test-model), and [score fusion](#ensemble-results).
 
 # Download dataset and preprocess
 We conduct experiments on two large multimodal action datasets, namely NTU RGB+D and NTU RGB+D 120. Download the dataset first, and then preprocess to generate mid-level features.
 
 ## Download dataset
 Request permission at RoseLab to download both datasets. [Link](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
 
 Download data for these modalities: Skeleton, Masked depth maps, RGB videos.
 
 # Build environment
 
 
 # Train and test model


 # Ensemble results

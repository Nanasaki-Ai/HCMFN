# HCMFN

**Human-Centric Multimodal Fusion Network for Robust Action Recognition**

# Abstract

The core idea of the proposed method is to address the challenge of effectively fusing different modalities in human action recognition. To overcome the presence of redundant background information, the method converts RGB, depth, and optical flow data into human-centric images (HCI) based on key point sequences. HCI prioritize the movement changes of essential body parts, minimizing the influence of unnecessary backgrounds and reducing computing resource requirements. Additionally, a human-centric multimodal fusion network (HCMFN) is introduced to learn action features from different modalities and extract comprehensive action patterns. 

This work has been submitted to **ESWA**. Please enlighten me with your instructions.

# Acknowledgements

 This work is based on the following two works：

 **MMNet**, TPAMI 2022, [Original code](https://github.com/bruceyo/MMNet)
 
 **PoseC3D**, CVPR 2022, [Original code](https://github.com/kennymckormick/pyskl)

 Thanks to the original authors for their work! Although our work represents only a modest improvement upon existing studies, we remain optimistic that it can provide valuable enlightenment to someone.

 Meanwhile, we are very grateful to the creators of these two datasets, i.e., NTU RGB+D and NTU RGB+D 120. Your selfless work has made a great contribution to the computer vision community!

 Last but not least, the authors will be very grateful for the selfless and constructive suggestions of the reviewers.

# Update notice
If you need to download on **Google Drive**, please contact me. Now you can also download everything on **Baidu Cloud**.

# Ablation studies

The detailed ablation studies on NTU RGB+D (CS/CV) and NTU RGB+D 120 (XSub/XSet) are as follows.

| Number | Input |  Backbone | CS(\%) | CV(\%) | XSub(\%) | XSet(\%) |
|--------|--------|--------|--------|--------|--------|--------|
| \#1    |  $J$                | PoseC3D        | 93.7 | 96.5 | 85.9 | 89.7 |
| \#2    |  $B$                | PoseC3D        | 93.4 | 96.0 | 85.9 | 89.7 |
| \#3    |  $J+B$              | PoseC3D        | 94.1 | 96.8 | 86.6 | 90.2 |
| \#4    |  $R$                | ResNet18       | 77.8 | 84.3 | 70.2 | 70.3 |
| \#5    |  $O$                | ResNet18       | 63.7 | 69.2 | 54.5 | 55.6 |
| \#6    |  $D$                | ResNet18       | 78.4 | 76.6 | 72.6 | 71.2 |
| \#7    |  $S+R$              | \#3+\#4        | 94.8 | 97.7 | 88.4 | 91.8 |
| \#8    |  $S+D$              | \#3+\#6        | 94.4 | 97.1 | 88.2 | 91.5 |
| \#9    |  $R+O$              | \#4+\#5        | 81.2 | 86.1 | 73.2 | 73.8 |
| \#10   |  $R+D$              | \#4+\#6        | 83.9 | 86.3 | 78.3 | 78.4 |
| \#11   |  $S+R+O$            | \#3+\#4+\#5    | 94.9 | 97.9 | 88.9 | 92.0 |
| \#12   |  $S+R+D$            | \#3+\#4+\#6    | 95.0 | 97.9 | 89.7 | 92.5 |
| \#13   |  $R+O+D$            | \#4+\#5+\#6    | 85.8 | 88.6 | 80.0 | 80.3 |
| \#14   |  $S+R+O+D$          | \#3+\#4+\#5+\#6| **95.1** | **98.0** | **89.9** | **92.7** |

$J$, $B$ represent 3D heatmap volumes of jonits and bones, respectively. $S$ stands for $J$ and $B$ together. $R$, $O$ and $D$ denote HCI of RGB, optical flow and depth, respectively. + indicates socre fusion. Bold accuracy indicates the best (HCMFN).

# How to reproduce this work?

 To reproduce this work, you must complete several stages, including

 **Step 1**, [Build environment](#build-environment)

 **Step 2**, [Download dataset and preprocess](#download-dataset-and-preprocess) 
  
 **Step 3**, [Train and test model](#train-and-test-model)
 
 **Step 4**, [Ensemble results](#ensemble-results)
 
 If you only need to quickly reproduce the experimental results in the article, please follow **Step 4**.

 # Build environment
 
 HCMFN has been enhanced by incorporating code from PoseC3D and MMNet. To evaluate its effectiveness in different contexts, we conduct experiments in various environments. Specifically, we utilize the environment of [MMLab](https://github.com/kennymckormick/pyskl) to process the input of 3D heatmap volumes, while the environment of [MMNet](https://github.com/bruceyo/MMNet) is employed for handling the input of HCI.

 # Download dataset and preprocess
 
 We conduct experiments on two large multimodal action datasets, namely NTU RGB+D and NTU RGB+D 120. Download the dataset first, and then preprocess to generate mid-level features.
 
 ## Download dataset
 
 Request permission at RoseLab to download both datasets. [Link](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
 
 Download data for these modalities: Skeleton, Masked depth maps, RGB videos.
 
 ## Preprocess
 
 The [3D heatmap volumes](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md) are available for download in PoseC3D. For convenience, we provide HCI for depth maps, RGB videos and optical flow modalities. Additionally, our code allows you to generate HCI for different modalities as per your requirements.
 
 **NTU RGB+D HCI**
 
 | HCI modalities | Baidu Cloud Link |  Google Drive Link |
 |----------------|------------------|--------------------| 
 | RGB HCI         | [Link](https://pan.baidu.com/s/1gogL--PS7UA26xmrKn52OA?pwd=t061) | - |
 | Optical flow HCI| [Link](https://pan.baidu.com/s/1evggahe3mbbilrMR5Y_gYg?pwd=ffce) | - |
 | Depth HCI       | [Link](https://pan.baidu.com/s/1mE2ZloU3tD4Y7l708xzpVg?pwd=1p98) | - |
 
 <p align="center">
  <img src="demo/rock-paper-scissors.gif" alt="rock-paper-scissors (RGB video)" width="400"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="demo/rock-paper-scissors-HCI.png" alt="rock-paper-scissors (RGB HCI)" width="400"/>
</p>

 <p align="center">
  <em>Rock-paper-scissors. Left: RGB video. Right: RGB HCI</em>
</p>

 <p align="center">
  <img src="demo/throw.gif" alt="rock-paper-scissors (RGB video)" width="400"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="demo/throw.png" alt="rock-paper-scissors (RGB HCI)" width="400"/>
</p>

 <p align="center">
  <em>Throw. Left: Masked depth maps. Right: Depth HCI</em>
</p>

 In view of the huge amount of data generated during framing, for each RGB video, we use the method of framing and deleting to construct RGB and optical flow HCI.
 
 Although NTU RGB+D has two benchmarks, note that we store all training samples and testing samples in the 'train' folder.
 
 Since we use flownet2 to generate optical flow HCI, you must download the pretrained model (173MB). [Baidu Cloud](https://pan.baidu.com/s/1WkT2e3O5RECTYxYeeQPKGQ?pwd=cr5w)

 For example, you can generate RGB and optical flow HCI:
 
    `python ntu60_gen_HCI.py`
    
 **Pay attention to modify the file path.**

| File path parameter    | Description         |
|------------------------|---------------------|
| skeletons_path         |  Raw skeleton data  |
| frames_path            |  Raw rgb video      |
| ignored_sample_path    |  Samples that need to be ignored (.txt) |
| out_folder             |  RGB HCI output file |
| out_folder_opt         |  Optical flow HCI output file |
    
 Additionally, you can generate depth HCI：
 
    `python ntu60_gen_depth_HCI.py`
 
 # Train and test model
 
 Each single-stream model is trained first, and then the learned model parameters are used for testing.
 
 Each data stream needs to be trained separately. For $J$ and $B$, please refer to the tutorial of PoseC3D. Here we introduce the training method about HCI.
 
 For RGB HCI:

    `python main.py`

We use the official code of [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) to generate labels for different benchmarks. You can also use these labels directly.

**Pay attention to modify the file path.**

| File path parameter    | Description         |
|------------------------|---------------------|
| data_path              |  Label file - default='data'                 |
| dataset                |  Label file - Dataset, i.e., ntu or ntu120   |
| dataset_type           |  Label file - Benchmark, i.e., xsub or xview |
| output                 |  Output file        |
| rgb_images_path        |  RGB HCI file       |

 For optical flow HCI:
 
    `python main_flow.py`
 
 For depth HCI:

    `python main_depth.py`

 # Ensemble results
 
 Perform weighted score fusion. Here we use the highest score finally obtained by each modality single-stream input for fusion. It is worth noting that the best results are not necessarily obtained from the fusion of these highest scores. Please try it yourself.
 
 **You can quickly reproduce the experimental results in the article based on the content of this part only.**

 Due to the upload file size limit (25MB), we store ensemble-related files (368MB) in [Baidu Cloud](https://pan.baidu.com/s/1mY2BWLJqxprsQ4cZB2-SEw).
 
 The files are arranged as follows:
 
         -ensemble\  
          -ntu60 
          -ntu120
          -ensemble60_xsub.py
          -ensemble60_xview.py
          -ensemble120_xset.py
          -ensemble120_xsub.py
          
 The four .py files correspond to the score fusion of the four benchmarks. You can change the alpha to adjust the weights for different modalities.
 
 For example, you can ensemble the results of the XSub, one of the benchmark of NTU RGB+D:
 
    `python ensemble60_xsub.py`
    
# Contact

If you find that the above description is not clear, or you have other issues that need to be communicated when conducting the experiment, please leave a message on Github.

Feel free to contact me via email:

    `zeshenghu@njnu.edu.cn`

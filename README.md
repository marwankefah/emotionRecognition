# Emotion-Recognition
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L-3uNEHGHFoBTAOwaHQJF38L2Gv-nwvl?usp=sharing)

Facial expression recognition system based on Improved Adaboost with Gabor Features

The execution time for processing 100 × 100 pixel size is 10 ms on CPU without CUDA-enabled GPU.


# Installation:
## Colab
Make Sure GPU is enabled to use CUDA-enabled GPU for processing
```
!pip install face_recognition
!git clone https://github.com/marwankefah/emotionRecognition
%cd emotionRecognition
!python run.py --imgPath [path to image]
```

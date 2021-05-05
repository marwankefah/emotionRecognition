# Emotion-Recognition
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L-3uNEHGHFoBTAOwaHQJF38L2Gv-nwvl?usp=sharing)

 Facial expression recognition system based on Improved Adaboost with Gabor Features.
# Summary
  - A Simple 2-layer neural network Improved AdaBoost based facial expression recognition system with Gabor Features. 
  - Average recognition rates in JAFFE, and CK+ is 98% and 97% respectively.
  - Supported Classes are Happy (0), Angry (1), Sad (2), Surprised (3), Neutral (4), Others (5).
  - The execution time for processing 100 × 100 pixel size is 10 ms on CPU.

### Ada.py
An Improved Adaboost Algorithm inspired from [Shen et al](https://link.springer.com/chapter/10.1007/11569947_6) to select best N features from Gabor Filters Bank that distinguish between Face (affectNet) / No Face (CIFAR-10) after balancing both datasets.  


## Prerequisites
 
- PyTorch 
- torchvision 
- NumPy
- face_recognition package (For Detection (can be replaced with any detection Algorithm)


## Run For a single Image
```
$ git clone https://github.com/marwankefah/emotionRecognition
```

```
$ python run.py --imgPath [path to image] --outPath [path to output]
```

### Example 1
```
$ python run.py --imgPath face.jpg --outPath result.jpg
```

![result](https://github.com/marwankefah/emotionRecognition/blob/master/result.jpg)
### Example 2
```
$ python run.py --imgPath surprised.jpg --outPath result1.jpg
```
![result1](https://github.com/marwankefah/emotionRecognition/blob/master/result1.jpg)
## Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L-3uNEHGHFoBTAOwaHQJF38L2Gv-nwvl?usp=sharing)
</br>
Make Sure GPU is enabled on COLAB to use CUDA-enabled GPU for processing
```
!pip install face_recognition
```
```
!git clone https://github.com/marwankefah/emotionRecognition
```
```
%cd emotionRecognition
```
```
!python run.py --imgPath [path to image] --outPath [path to output]
```

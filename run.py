import argparse
from datetime import datetime

import face_recognition
import matplotlib.pyplot as plt

import src.Helper as hf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emotion Recognition')
    parser.add_argument("--imgPath",
                        help="Image To get the Emotion Recognition For",
                        type=str)
    args = parser.parse_args()
    real, imag = hf.build_filters()
    #Path Of Trained Model
    modelPath = "D:\\marwan\\Masters\\Bath\\emotionRecgonition\\model\\checkpoint.pth"
    # CSV file used to obtain the most important gabor features
    path2 = "D:\\marwan\\Masters\\Bath\\emotionRecgonition\\a.csv"

    # imgPath = "D:\\marwan\\Masters\\Bath\\emotionRecgonition\\face.jpg"
    imgPath=args.imgPath
    image = face_recognition.load_image_file(imgPath)

    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        face = image[face_location[0]:face_location[2], face_location[3]: face_location[1]]
        emotion1, prob1 = hf.getemotion(face, modelPath, path2)
        plt.imshow(face)
        plt.title("Emotion is {0} with probability = {1:.3f}% ".format(emotion1, float(prob1[0][0])))
        plt.show()

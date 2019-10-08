import imutils
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

model = load_model("hgrf_model.h5")
bg = None
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg

    if bg is None:
        bg = image.copy().astype('float')
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def getPredictedClass():
        # Predict
        image = cv2.imread('Temp.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(320,120))
        image = np.array(image, dtype="uint8")
        image = image.reshape(1, 120, 320, 1)
        prediction = model.predict(image)
        return np.argmax(prediction), (np.argmax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))


def showStatistics(predictedClass, confidence):

        textImage = np.zeros((300, 512, 3), np.uint8)
        class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
        className = class_names[predictedClass]


        cv2.putText(textImage, "Pedicted Class : " + className,
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

        cv2.putText(textImage, "Confidence : " + str(confidence * 100) + '%',
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
        cv2.imshow("Statistics", textImage)



def segment(image, threshold=25):
    global bg

    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:

        segmented = max(cnts, key=cv2.contourArea)
        return thresholded, segmented


def main():
    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 255, 590

    num_frames = 0
    start_recording = False

    while(True):

        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)


        if num_frames < 30:
            run_avg(gray, aWeight)
        else:

            hand = segment(gray)

            if hand is not None:

                (thresholded, segmented) = hand

                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:

                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)

        num_frames += 1

        cv2.imshow('Video Feed', clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord('q'):
            break
        if keypress == ord("s"):
            start_recording = True

main()
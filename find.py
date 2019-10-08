

import cv2
import imutils
import numpy as np
from keras.models import load_model
from PIL import Image
from actions import data_entry, email, sendSMS


model = load_model("hgr_model1.h5")

# global variables
bg = None


def run_avg(image, aWeight):
    global bg

    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(imageName)


def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100, 116))
    image = np.array(image, dtype="uint8")
    image = image.reshape(1, 116, 100, 1)
    prediction = model.predict(image)
    print(np.argmax(prediction))
    # print(np.amax(prediction) / prediction.sum())

    return np.argmax(prediction), (np.amax(prediction) / prediction.sum())


def action_counter(pc):
    global k, l, u, t, p
    class_names = ["K", "L", "P", "T", "U"]
    # n = class_names[pc - 1]
    if pc == 0:
        k += 1
    elif pc == 1:
        l += 1
    elif pc == 2:
        p += 1
    elif pc == 3:
        t += 1
    elif pc == 4:
        u += 1
    print((k, l, p, t, u))
    return (k, l, p, t, u)


def action_master(cls):
    status = ''
    global ty
    if cls == 0:  # k
        status += email('K')
    elif cls == 1:  # l
        status += email('L')
    elif cls == 2:  # p
        status += data_entry()
    elif cls == 3:  # t
        status += sendSMS()

    # elif cls ==4: #u
    return status



def segment(image, threshold=25):
    global bg

    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:

        segmented = max(cnts, key=cv2.contourArea)
        return thresholded, segmented

def preds(thresholded):
    cv2.imwrite('Temp.png', thresholded)
    resizeImage('Temp.png')
    return getPredictedClass()



k, l, p, t, u = 0, 0, 0, 0, 0
da = 0

def main():
    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0
    image_num = 0

    start_recording = False

    while True:

        (grabbed, frame) = camera.read()
        if grabbed:

            frame = imutils.resize(frame, width=700)

            frame = cv2.flip(frame, 1)

            clone = frame.copy()

            (height, width) = frame.shape[:2]

            roi = frame[top:bottom, right:left]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if num_frames < 30:
                run_avg(gray, aWeight)
                print(num_frames)
            else:

                hand = segment(gray)

                if hand is not None:

                    (thresholded, segmented) = hand

                    dong = cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    textImage = np.zeros((300, 512, 3), np.uint8)
                    if start_recording:
                        # cv2.imwrite("Dataset/FistTest/fist_" + str(image_num) + '.png', thresholded)
                        predictedClass, confidence = preds(thresholded)
                        class_names = ["K", "L", "P", "T", "U"]
                        n = action_counter(predictedClass)
                        cv2.putText(clone, str(class_names[predictedClass - 1]) + ' ' + str(confidence * 100) + '%',
                                    (300, 300), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255),
                                    2)

                        global k, l, p, t, u
                        if max(n) >= 100:
                            k, l, p, t, u = 0,0,0,0,0
                            cv2.putText(textImage, str(class_names[n.index(max(n)) - 1]) + ' Initiating Action......',
                                        (0, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255),
                                        2)




                            if operation_on == True:
                                start_recording = False

                                # nala = cv2.namedWindow('actions', cv2.WINDOW_NORMAL)

                                cv2.imwrite('Alert.png', clone)
                                cv2.imwrite('Record.png', clone)


                                # cv2.putText(textImage, str(class_names[n.index(max(n)) - 1]) + ' Initiating Action......',
                                #             (0, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255),
                                #             2)



                                if action_master(predictedClass - 1) == 'Successful':
                                    cv2.putText(textImage,
                                                str(class_names[n.index(max(n)) - 1]) + ' Action Completed!......',
                                                ( 100, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255),
                                                2)
                                elif action_master(predictedClass - 1) == 'Unsuccessful':
                                    cv2.putText(textImage,
                                                str(class_names[n.index(max(n)) - 1]) + ' Action Aborted......',
                                                (200, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255),
                                                2)

                                cv2.imshow("Actions", textImage)
                                k, l, p, t, u = 0, 0, 0, 0, 0

                        # showStatistics(predictedClass, confidence)

                        image_num += 1
                    cv2.imshow("Threshold", thresholded)

            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            num_frames += 1

            cv2.imshow("Video Feed", clone)

            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord("q"):
                break

            if keypress == ord("s"):
                start_recording = True
                operation_on = True

            if keypress == ord("z"):
                start_recording = True
                operation_on = False


        else:
            print("[Warning!] Error input, Please check your(camera Or video)")
            break


main()

from PIL import Image
# organize imports
import cv2
import imutils
import numpy as np
import os
# global variables
bg = None

def data_number():

    le = ["K", "L", "P", "T", "U"]
    dn = []
    for i in le:
        file_list = sorted(os.listdir("/home/cipher/PycharmProjects/hand_gesture/Dataset/" +i))

        dn.append(int(len(file_list) - 1))
    return dn




# for i in range(0, 1000):

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

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
        return (thresholded, segmented)

if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5
    exam = data_number()
    kn, ln, pn, tn, un = int(exam[0]), int(exam[1]), int(exam[2]), int(exam[3]), int(exam[4])

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 300, 590

    # initialize num of frames
    num_frames = 0
    # counter = 2001
    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                cv2.imshow("Thesholded", thresholded)
                x, y, w, h = cv2.boundingRect(segmented)
                outlined_image = cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # cv2.putText(outlined_image, 'pp', cv2.FONT_HERSHEY_SIMPLEX, 7, (100, 255, 100), 2)
                cv2.putText(outlined_image, "Pedicted Class : ", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255),
                            2)

                # img_counter += 1

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        # kn,ln,pn,tn,un = exam[0],exam[1],exam[2],exam[3],exam[4]

        if keypress == ord('k'):
            kn += 1
            working_location = os.chdir("/home/cipher/PycharmProjects/hand_gesture/Dataset/K")
            working_location = os.getcwd()
            cv2.imwrite('1' + '_' + 'K'+ '_' +str(kn)+ '.png', thresholded)
            print(str(kn) + 'k')
        if keypress == ord('l'):
            ln += 1
            working_location = os.chdir("/home/cipher/PycharmProjects/hand_gesture/Dataset/L")
            working_location = os.getcwd()
            cv2.imwrite('2' + '_' + 'L'+ '_' +str(ln)+ '.png', thresholded)
            print(str(ln) + 'l')

        if keypress == ord('p'):
            pn += 1
            working_location = os.chdir("/home/cipher/PycharmProjects/hand_gesture/Dataset/P")
            working_location = os.getcwd()
            cv2.imwrite('3' + '_' + 'P'+ '_' +str(pn)+ '.png', thresholded)
            print(str(pn) + 'p')

        if keypress == ord('t'):
            tn += 1
            working_location = os.chdir("/home/cipher/PycharmProjects/hand_gesture/Dataset/T")
            working_location = os.getcwd()
            cv2.imwrite('4' + '_' + 'T'+ '_' +str(tn)+ '.png', thresholded)
            print(str(tn) + 't')

        if keypress == ord('u'):
            un += 1
            working_location = os.chdir("/home/cipher/PycharmProjects/hand_gesture/Dataset/U")
            working_location = os.getcwd()
            cv2.imwrite('5' + '_' + 'U'+ '_' +str(un)+ '.png', thresholded)
            print(str(un) + 'u')


        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()

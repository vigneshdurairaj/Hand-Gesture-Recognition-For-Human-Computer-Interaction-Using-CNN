import os
import cv2







def resizeImage(imageName):
    le = ["K", "L", "P", "T", "U"]

    for j in le:
        for i in range(0,7001):

            Image = cv2.imread("/home/cipher/PycharmProjects/hand_gesture/" + j +"/" + j + '0' + '.png')
            basewidth = 100
            img = Image.open(imageName)
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.save(imageName)
            print('p')


for i in range(0, 3001):
    print(i)
    # Mention the directory in which you wanna resize the images followed by the image name
    resizeImage("/home/cipher/PycharmProjects/hand_gesture/P/P" + str(i) + '.png')

def rename():
    le = ["K", "L", "P", "T", "U"]

    for i in le:
        path = '/home/cipher/PycharmProjects/hand_gesture/Dataset1/'+ i
        j = 0
        for filename in os.listdir(path):
            os.rename(os.path.join(path,filename), os.path.join(path,str(le.index(i) + 1)+ '_' + i + '_' +str(j)+ '.png'))
            j = j +1
        print(j)
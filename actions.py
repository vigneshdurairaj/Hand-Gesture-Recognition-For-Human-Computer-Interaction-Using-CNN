import datetime
import os
import smtplib
import ssl
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from oauth2client.file import Storage
from oauth2client import client
from oauth2client import tools

import httplib2
from googleapiclient import discovery

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

from werkzeug.wrappers import json


def email(option):
    t = ''
    try:

        sender_email = "xxx@gmail.com"
        receiver_email = "yyy@gmail.com"
        password = "d########"
        ImgFileName = 'Alert.png'

        message = MIMEMultipart("alternative")
        message["Subject"] = "HGR_Message"
        message["From"] = sender_email
        message["To"] = receiver_email
        img_data = open(ImgFileName, 'rb').read()

        # Create the plain-text and HTML version of your message
        text = """\
        Hi,
        This an automated message
        From HGR_Systems 
        Alert! Alert! Alert!"""
        image = MIMEImage(img_data, name=os.path.basename(ImgFileName))

        # Turn these into plain/html MIMEText objects
        part1 = MIMEText(text, "plain")
        # part2 = MIMEText(html, "html")

        # Add HTML/plain-text parts to MIMEMultipart message
        # The email client will try to render the last part first
        if option == 'K':
            message.attach(part1)

        elif option == 'L':
            message.attach(part1)
            message.attach(image)
        # message.attach(part2)

        # Create secure connection with server and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(
                sender_email, receiver_email, message.as_string()
            )
        t += 'Successful'
    except:
        t += 'Unsuccessful'

    return t


def data_entry():
    t = ''
    file1 = open("data.txt", "a")
    str1 = '   A vehicle entered at' + ' ' + str(datetime.datetime.now())
    file1.write(str1)
    file1.close()
    t = 'Successful'
    return t


def data_upload(time):
    global ty
    t = 0
    print('before try')
    try:


        t = 'Successful'
    except:
        t = 'Unsuccessful'


import time
from time import sleep
from sinchsms import SinchSMS


# function for sending SMS
def sendSMS():


    number = '+91999*******'
    message = 'This is an automated message from HGR_System! You"ve been sent an alert from the registered person. '

    client = SinchSMS( '<Give your SINCH KEY HERE>' )

    print("Sending '%s' to %s" % (message, number))
    response = client.send_message(number, message)
    message_id = response['messageId']

    response = client.check_status(message_id)
    while response['status'] != 'Successful':
        print(response['status'])
        time.sleep(1)
        response = client.check_status(message_id)
        print(response['status'])
    return 'Successful'

# if __name__ == '__main__':



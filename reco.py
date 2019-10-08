from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
def data_upload():
    t = 0
    print('before try')
    try:
        print('befior create file')
        file1 = drive.CreateFile()
        print('after create file')
        file1.SetContentFile('Record.png')
        print('after records file')

        file1.Upload()
        print('--------Uploaded---------------')
        t = 'Successful'
    except:
        t = 'Unsuccessful'


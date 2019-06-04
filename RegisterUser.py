import cv2
import os

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class User:

    def __init__(self,user_name):
        self.user_name = user_name

    def getUserName(self):
        return self.user_name

    @staticmethod
    def face_extractor(img):
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(img, 1.3, 10)
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    def register_user(self):
        cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cap.read()
            if self.face_extractor(frame) is not None:
                count = count + 1
                face = cv2.resize(self.face_extractor(frame), (400, 400))
                # face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                file_name_path = 'faces\\' + self.getUserName() + '/' + 'sample' + str(count) + '.jpg'
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', face)
            else:
                print('Face not Found')
            if cv2.waitKey(1) == 13 or count == 100:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Collecting Samples Complete!!!')



while True:
    choice = input("Wanted to Register:")
    if choice == "yes":
       try:
           name = input("Enter your name")
           usr = User(name)
           os.mkdir('faces/'+usr.getUserName())
           usr.register_user()
       except Exception as e:
           print(e)
           print("This user already exists")
    else:
        break









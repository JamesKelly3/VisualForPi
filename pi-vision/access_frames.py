import cv2
import time
import numpy as np
import dill as pickle
import os

frame_no = 0
try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
except:
    picamera = False
    print("Could not import picamera")


def main():
    # faces, labels = pickle.load(open('faces.p', 'rb')) if os.path.isfile('faces.p') else [], []
    faces, labels = [], []
    fgbg = cv2.createBackgroundSubtractorMOG2()
    if picamera:
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(640, 480))

        # allow the camera to warmup
        time.sleep(0.1)
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
           # image = fgbg.apply(image)
            process(image, faces, labels)
    else:
        cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_BRIGHTNESS, 55)
        # cap.set(cv2.CAP_PROP_GAIN, 25)
        while True:
            ret, frame = cap.read()
           # frame = fgbg.apply(frame)
            process(frame, faces, labels)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


def process_face(face, i, faces, labels):
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    face = cv2.fastNlMeansDenoising(face, None, 6, 15, 5)
    face = cv2.Canny(face, 25, 15)
    # kernel = np.ones((5, 5), np.uint8)
    connectivity = 15
    face, contours, hierarchy = cv2.findContours(face, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    output = cv2.connectedComponents(face, connectivity, cv2.CV_32S)
    face[output[1] == 0] = 255
    face[output[1] != 0] = 0
    if frame_no % 100 == 0:
        faces.append(face)
        labels.append("James")
       # pickle.dump((faces, labels), open('faces.p', 'wb'))
    cv2.imshow("Face" + str(i), face)


def process(frame, faces2, labels):
    frame = cv2.flip(frame, 1)
    faces = face_cascade.detectMultiScale(frame, 1.1, 8)
    for i in range(len(faces)):
        x, y, w, h = faces[i]
        cv2.rectangle(frame, (x + int(0.075 * w), y), (x + int(0.075 * w) + int(0.85 * w), y + h), (255, 0, 0), 2)
        face = frame[y:y + h, x + int(0.075 * w):x + int(0.075 * w) + int(0.85 * w)]
        face = cv2.resize(face, (250, 250))
        process_face(face, i, faces2, labels)
        # cv2.imshow("face", face)
    # process_face(frame, 0)
    cv2.imshow("Original", frame)


if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
    main()

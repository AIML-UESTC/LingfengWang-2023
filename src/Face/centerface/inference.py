import cv2
import sys
from src.Face.centerface.centerface import CenterFace
import matplotlib.pyplot as plt

def load_model():
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    return centerface

def detect_faces(centerface,frame):
    h, w = frame.shape[:2]
    dets, lms = centerface(frame, h, w, threshold=0.35)
    return dets,lms

def draw(frame,dets,lms):
    for det in dets:
        #boxes, score = det[:4], det[4]
        boxes = det[:4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 2)
    for lm in lms:
        for i in range(0, 5):
            cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def test_image(img_path):
    """[Face detection]

    Args:
        img_path ([str]): [path to the image]
    """
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
    else:
        dets = centerface(frame, threshold=0.35)

    for det in dets:
        #boxes, score = det[:4], det[4]
        boxes = det[:4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 2)
    if landmarks:
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(32,32))
    plt.imshow(frame)
    #plt.savefig('result_' + img_path, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    image_path = sys.argv[1]
    test_image(image_path)

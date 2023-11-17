'''Demo of MTCNN face detector on Image

$ python3 trt_mtcnn_image.py --image test_face.jpg

'''

import argparse
import cv2
from utils.mtcnn import TrtMtcnn


WINDOW_NAME = 'TrtMtcnnDemo'
BBOX_COLOR = (0, 255, 0)  # green


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    parser.add_argument('--image', type=str, default='test_face.jpg',
                        help='input image for testing of mtcnn')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        for j in range(5):
            cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, BBOX_COLOR, 2)
    return img

def main():
    args = parse_args()

    image = cv2.imread(args.image)

    mtcnn = TrtMtcnn()

    if image is not None:
        dets, landmarks = mtcnn.detect(image, minsize=args.minsize)
        print('{} face(s) found'.format(len(dets)))
        image = show_faces(image, dets, landmarks)
        cv2.imwrite("output_mtcnn.jpg", image)

if __name__ == '__main__':
    main()

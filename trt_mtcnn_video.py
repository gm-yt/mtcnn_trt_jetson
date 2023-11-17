'''Demo of MTCNN face detector on Video

$ python3 trt_mtcnn_video.py --video test_face.mp4

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
    parser.add_argument('--video', type=str, required=True,
                        help='input video for testing of mtcnn')
    parser.add_argument('--save', type=bool, default=True,
                        help='save or not save the output video')
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

    mtcnn = TrtMtcnn()

    cap = cv2.VideoCapture(args.video)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.save == True:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_mtcnn.mp4',fourcc, 30.0, (video_width,video_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dets, landmarks = mtcnn.detect(frame, minsize=args.minsize)
        print('{} face(s) found'.format(len(dets)))
        frame = show_faces(frame, dets, landmarks)

        if args.save == True:
            # write the frame
            out.write(frame)

    if args.save == True:
        out.release()
    cap.release()

if __name__ == '__main__':
    main()

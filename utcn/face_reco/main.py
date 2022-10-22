import argparse
import typing
import cv2
import pathlib
import os


models = {
        'face': cv2.data.haarcascades + 'haarcascade_profileface.xml',
        'eyes': cv2.data.haarcascades + 'haarcascade_eye.xml',
        'lips': os.path.join(os.path.abspath(os.path.dirname(__file__)), 'mouth.xml')
        # 'lips': './mouth.xml'
        }

print(models['lips'])

config = {
        'face': {
            'scaleFactor': 1.1,
            'minNeighbors': 10,
            'minSize': (50, 50)
            },
        'eyes': { 
           'scaleFactor':1.1,
            'minNeighbors': 40,
            'minSize': (40, 40)
              },
        'lips': {
            'scaleFactor': 1.1,
            'minNeighbors': 40,
            'minSize': (20, 20)
           }
        }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', action='store_true', help='Use internal camera as input.')
    parser.add_argument('--face-part', choices=['face', 'eyes', 'lips'], default='face', help = 'Choose a face component.')
    return parser


def face_detection(camera_input: bool = False, face_part_choice: str = 'face') -> int:
    if not camera_input:
        return 0
    clf = cv2.CascadeClassifier(models[face_part_choice])
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = clf.detectMultiScale(
                gray,
                **config[face_part_choice],
                flags = cv2.CASCADE_SCALE_IMAGE
            )
        for (x, y, width, height) in detections:
            cv2.rectangle(frame, (x, y), (x + width, y + width), (255, 255, 0), 2)
        cv2.imshow('faces', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    face_detection(
           camera_input = args.camera,
           face_part_choice = args.face_part
           )
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

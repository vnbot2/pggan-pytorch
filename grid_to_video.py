from common import *
from pyson.vision import put_text
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i')
parser.add_argument('--output_path', '-o')
parser.add_argument('--fps', default=5, type=int)
parser.add_argument('--size', default=5)
args = parser.parse_args()



frame_array = []
image_paths = glob.glob(args.input_dir+'/*.jpg')
def sort_fn(x):
    return int(os.path.basename(x).split('_')[0])

image_paths = list(sorted(image_paths, key=sort_fn))

size = None
def f(image_path):
    global size
    img = cv2.imread(image_path)
    height, width, layers = img.shape
    size = (width,height)
    put_text(img, (50, 50), str(sort_fn(image_path)))
    return img

frame_array = multi_thread(f, image_paths, verbose=True)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(args.output_path, fourcc, args.fps, size)


for frame in frame_array:
    # writing to a image array
    out.write(frame)
out.release()
print('Save at:', args.output_path)

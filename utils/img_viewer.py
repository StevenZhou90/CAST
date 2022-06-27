"""
Use this script to view all images from an image list
E.g:
python img_viewer.py --num=3 --time=1 --imglist ../validation_sets/set1.list
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import os
from os.path import basename, dirname, join
import argparse

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, help='number imgs on screen', default=3)
parser.add_argument('--time', type=float, help='time on screen', default=1.)
parser.add_argument('--imglist', type=str, help='path to img list', default='/home/wes/Data/face/data/celebA/anno/img.list')
args = parser.parse_args()
args.pad=.1


def get_imglist(args):
    with open(args.imglist, 'r') as f:
        x = f.readlines()
    x = [a.strip() for a in x]
    return x

def get_plot_shape(args):
    # if args.num > 3:
    #     return 2, args.num/2
    # else:
    return 1, args.num

imglist = get_imglist(args)
rows, cols = get_plot_shape(args)

# matplotlib set up
plt.style.use('dark_background')
mpl.rcParams['toolbar'] = 'None'


fig, ax = plt.subplots(rows, cols)
fig.tight_layout(w_pad=args.pad, h_pad=args.pad, pad=max(args.pad, 0))
fig.canvas.toolbar_visible = False
fig.canvas.header_visible = False
fig.canvas.footer_visible = False
fig.set_snap(True)
manager =  plt.get_current_fig_manager()
x_pos = 2000*1 + 300
manager.window.geometry(f'500x500+{x_pos}+300')
manager.full_screen_toggle()


idx = 0

# loop through img list
num = rows*cols
while True:
    imgs_ = []
    for j in range(num):
        img_idx = idx + j
        pth = imglist[img_idx]
        # imgs_.append(join(basename(dirname(pth)), basename(pth)))
        image = Image.open(pth)
        # box = bbox[img_idx]
        # crop = get_crop(ldmk[idx+j], *image.size)
        # h,w = image.size
        # crop = expand_crop(h,w, box, factor=1.35)
        # image = image.crop(crop)
        # image = image.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))

        if rows == 1:
            ax[j].imshow(image)
            ax[j].axis('off')
        else:
            r = int(j / cols)
            c = j % cols
            ax[r][c].imshow(image)
            ax[r][c].axis('off')


    plt.draw()
    plt.pause(args.time)
    for a in fig.axes:
        plt.sca(a)
        plt.cla()
        plt.axis('off')

    idx += num

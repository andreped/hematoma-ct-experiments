import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import SimpleITK as sitk
import cv2
from matplotlib.widgets import Slider
from tqdm import tqdm
from skimage.morphology import binary_opening, disk, ball, remove_small_holes, remove_small_objects, binary_dilation
from numpy.random import shuffle
import os
import h5py
import cv2
from scipy.ndimage import binary_fill_holes

def import_set(tmp, file):
    f = h5py.File(file, 'r')
    tmp = np.array(f[tmp])
    f.close()
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    return tmp

def unique_patients(tmp):
    l = []
    for t in tmp:
        l.append(int(t.split("/")[-2]))
    return np.unique(l)

def images(event):
    ax[0].clear()
    ax[0].imshow(data[int(slider2.val)], cmap='gray')
    ax[0].imshow(gt[int(slider2.val)], cmap=cmap, alpha=float(slider1.val))
    ax[0].imshow(gt_b[int(slider2.val)], cmap=cmap2)
    ax[0].set_title('CT + GT')
    ax[0].set_axis_off()

    ax[1].clear()
    ax[1].imshow(out[int(slider2.val)], cmap='gray')
    ax[1].set_title('Skull stripping')
    ax[1].set_axis_off()

    f.suptitle('slice ' + str(int(slider2.val)))
    f.canvas.draw_idle()


def up_scroll_alt(event):
    if event.key == "up":
        if (slider2.val + 2 > data.shape[0]):
            1
        # print("Whoops, end of stack", print(slider2.val))
        else:
            slider2.set_val(slider2.val + 1)


def down_scroll_alt(event):
    if event.key == "down":
        if (slider2.val - 1 < 0):
            1
        # print("Whoops, end of stack", print(slider2.val))
        else:
            slider2.set_val(slider2.val - 1)


def up_scroll(event):
    if event.button == 'up':
        if (slider2.val + 2 > data.shape[0]):
            1
        # print("Whoops, end of stack", print(slider2.val))
        else:
            slider2.set_val(slider2.val + 1)


def down_scroll(event):
    if event.button == 'down':
        if (slider2.val - 1 < 0):
            1
        # print("Whoops, end of stack", print(slider2.val))
        else:
            slider2.set_val(slider2.val - 1)


if __name__ == "__main__":

    data_loc = "/home/andrep/workspace/hematoma/data/"
    gt_loc = "/home/andrep/workspace/hematoma/gt/"

    gts = os.listdir(gt_loc)
    shuffle(gts)

    for name in tqdm(gts, "CT:"):

        print(name)

        locs = data_loc + str(name) + ".nii.gz"
        gt_locs = gt_loc + str(name) + "/" + str(name) + "-label.nrrd"

        itkimage = sitk.ReadImage(locs)
        data = sitk.GetArrayFromImage(itkimage)

        itkimage = sitk.ReadImage(gt_locs)
        gt = sitk.GetArrayFromImage(itkimage)

        # fix orientation
        data = np.rot90(data, k=2, axes=(1, 2))
        data = np.flip(data, axis=2)
        gt = np.rot90(gt, k=2, axes=(1, 2))
        gt = np.flip(gt, axis=2)

        data_orig = data.copy()

        print(data.shape)
        print(gt.shape)

        # HU clipping
        limits = (0, 100)
        data[data < limits[0]] = limits[0]
        data[data > limits[1]] = limits[1]
        data = data / np.amax(data) * 255

        #plt.imshow(data[150], cmap="gray")
        #plt.show()

        #exit()

        # skull stripping
        # choose method
        method = 3

        if method == 1:
            #data_orig[data_orig > 230] = 0
            data_orig = data_orig / np.amax(data_orig) * 255
            data = data.astype(np.uint8)

            out = np.zeros_like(data).astype(np.float32)
            for i in tqdm(range(data.shape[0])):
                tmp = data[i]
                if np.count_nonzero(tmp) == 0:
                    out[i] = data[i]
                else:
                    #plt.imshow(data[i], cmap="gray")
                    #plt.show()
                    ret, thresh = cv2.threshold(data[i], 0, 255, cv2.THRESH_OTSU)
                    ret, markers = cv2.connectedComponents(thresh)
                    # Get the area taken by each component. Ignore label 0 since this is the background.
                    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
                    if marker_area == []:
                        out[i] = data[i]
                    else:
                        # Get label of largest component by area
                        largest_component = np.argmax(marker_area) + 1  # Add 1 since we dropped zero above
                        # Get pixels which correspond to the brain
                        brain_mask = markers == largest_component
                        brain_out = data[i].copy()
                        brain_out[brain_mask == False] = 0
                        out[i] = brain_out

        elif method == 2:
            out = data_orig.copy()
            th = 900 # HU for bone [1800, 1900]
            out[out <= th] = 0
            out[out > th] = 1

        elif method == 3:
            out = data_orig.copy()
            limits = (0, 100)
            out[out < limits[0]] = limits[0]
            out_orig = out.copy()
            out[out > limits[1]] = limits[0]
            out_orig[out_orig > limits[1]] = limits[1]
            tmp = out.copy()
            tmp = cv2.medianBlur(tmp, 3)
            tmp[tmp > 0] = 1
            tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, disk(11))
            tmp = remove_small_objects(tmp.astype(bool), min_size=0.1*np.prod(tmp.shape)).astype(int)
            tmps = tmp.copy()
            for i in range(tmps.shape[0]):
                tmps[i] = remove_small_holes(tmps[i].astype(bool), area_threshold=0.1*np.prod(tmps[i].shape)).astype(int)
                tmps[i] = cv2.dilate(tmps[i].astype(np.uint8), disk(7))
                tmps[i] = cv2.erode(tmps[i].astype(np.uint8), disk(7))
                tmps[i] = binary_fill_holes(tmps[i])
            #tmps = binary_dilation(tmps, ball(9))
            #tmps = cv2.morphologyEx(tmps.astype(np.uint8), cv2.MORPH_CLOSE, disk(7))
            out_orig[tmps != 1] = 0
            out = out_orig.copy()
            #out = tmps.copy()
            gt = tmps.copy()

        else:
            print('test')
            out = data.copy()


        # generate boundary image
        gt_b = np.zeros_like(gt)
        for i in range(gt.shape[0]):
            if len(np.unique(gt[i])) > 1:
                gt_b[i] = cv2.Canny((gt[i] * 255).astype(np.uint8), 0, 255)

        gt_b = gt_b.astype(np.float32)
        gt_b = gt_b / np.amax(gt_b)

        colors = [(0, 0, 1, i) for i in np.linspace(0, 1, 3)]
        cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
        colors = [(1, 0, 0, i) for i in np.linspace(0, 1, 3)]
        cmap2 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

        f, ax = plt.subplots(1, 2, figsize=(12, 12))
        f.canvas.mpl_connect('key_press_event', up_scroll_alt)
        f.canvas.mpl_connect('key_press_event', down_scroll_alt)
        f.canvas.mpl_connect('scroll_event', up_scroll)
        f.canvas.mpl_connect('scroll_event', down_scroll)

        s1ax = plt.axes([0.25, 0.08, 0.5, 0.03])
        slider1 = Slider(s1ax, 'alpha', 0, 1.0, dragging=True, valstep=0.05)

        s2ax = plt.axes([0.25, 0.02, 0.5, 0.03])
        slider2 = Slider(s2ax, 'slice', 0, data.shape[0] - 1, valstep=1, valfmt='%1d')

        # init
        slider1.set_val(0.3)
        slider2.set_val(0)
        f.subplots_adjust(bottom=0.15)

        slider1.on_changed(images)
        slider2.on_changed(images)
        slider2.set_val(slider2.val)

        plt.show()

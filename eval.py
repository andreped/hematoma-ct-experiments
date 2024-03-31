import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import SimpleITK as sitk
import cv2
from matplotlib.widgets import Slider
from tqdm import tqdm
from tensorflow.keras.models import load_model
import os
import h5py
from numpy.random import shuffle
from scipy.ndimage.interpolation import zoom
from skimage.morphology import binary_opening, disk, ball, remove_small_holes, remove_small_objects, binary_dilation
from scipy.ndimage import binary_fill_holes


def DSC(pred, gt):

    smooth = 0 # <- should be 0 when doing inference(!)

    output1 = pred.copy()
    target1 = gt.copy()

    intersection1 = np.sum(output1 * target1)
    union1 = np.sum(output1 * output1) + np.sum(target1 * target1)
    dice = (2. * intersection1 + smooth) / (union1 + smooth)

    return dice

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


def threshold(event):
    global pred_b, pred_tmp

    pred_tmp = pred.copy()

    th = slider3.val
    print(th)
    if th == 0:
        pred_tmp = np.zeros_like(pred)
        pred_b = np.zeros_like(pred_tmp)
    else:
        pred_tmp[pred_tmp < th] = 0
        pred_tmp[pred_tmp >= th] = 1

    if np.count_nonzero(pred_tmp) == 0:
        pred_b = np.zeros_like(pred_tmp)
    else:
        # generate boundary image
        pred_b = np.zeros_like(pred_tmp)
        for i in range(pred_tmp.shape[0]):
            if np.count_nonzero(pred[i]) != 0:
                pred_b[i] = cv2.Canny((pred_tmp[i] * 255).astype(np.uint8), 0, 255)

        pred_b = pred_b.astype(np.float32)
        pred_b = pred_b / np.amax(pred_b)

    slider2.set_val(slider2.val)


def images(event):
    ax[0].clear()
    ax[0].imshow(data_orig[int(slider2.val)], cmap='gray')
    ax[0].imshow(gt[int(slider2.val)], cmap=cmap, alpha=float(slider1.val))
    ax[0].imshow(gt_b[int(slider2.val)], cmap=cmap2)
    ax[0].imshow(pred_b[int(slider2.val)], cmap=cmap3)
    ax[0].imshow(brain[int(slider2.val)], cmap=cmap4, alpha=float(slider4.val))
    ax[0].imshow(brain_b[int(slider2.val)], cmap=cmap5)
    ax[0].set_title('CT + GT + PRED + BRAIN')
    ax[0].set_axis_off()

    ax[1].clear()
    ax[1].imshow(gt[int(slider2.val)], cmap='gray', vmin=0, vmax=1)
    ax[1].set_title('GT')
    ax[1].set_axis_off()

    ax[2].clear()
    ax[2].imshow(pred_tmp[int(slider2.val)], cmap='gray')
    ax[2].set_title('Pred' + ', DSC: ' + str(np.round(dsc_list[int(slider2.val)], 3)))
    ax[2].set_axis_off()

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


def predict(input_volume_path, output_mask_path, out_dim):

    print("Extracting data...")
    itkimage = sitk.ReadImage(input_volume_path)
    data = sitk.GetArrayFromImage(itkimage)

    print(np.unique(data))

    print("Pre-processing...")
    # fix orientation
    data = np.rot90(data, k=2, axes=(1, 2))
    data = np.flip(data, axis=2)

    # HU-clip
    limits = (0, 140)
    data[data < limits[0]] = limits[0]
    data[data > limits[1]] = limits[1]

    # scale -> [0, 1]
    data = data - np.amin(data)
    data = data / np.amax(data)
    # data = np.round(data)

    # get resolution information
    img_size = out_dim[1:]
    tmp1 = itkimage.GetSpacing()
    tmp2 = itkimage.GetSize()
    res_factor = tmp1[2] / tmp1[0]
    res = np.ones(3)
    res[0] = res_factor
    res[1] = img_size[0] / tmp2[0]
    res[2] = img_size[1] / tmp2[1]

    # interpolate, (Nx512x512, with slice thickness 1)
    data = zoom(data, zoom=res, order=1)
    data = data - np.amin(data)
    data = data / np.amax(data)

    # model path
    # model_path = input_volume_path.split("/")[:-2]
    # model_path = "/home/deepinfer/github/hematoma-segmenter/hematomasegmenter/nets/unet_model.h5" #"/".join(model_path) + "/nets/" + "unet_model.h5"

    print("Loading model...")
    # load trained model
    model = load_model(model_path, compile=False)

    print("Predicting...")

    # split data into chunks and predict
    out_dim = (16, 512, 512)
    preds = np.zeros((int(np.ceil(data.shape[0] / out_dim[0])),) + out_dim + (2,)).astype(np.float32)
    for i in range(int(np.ceil(data.shape[0] / out_dim[0]))):
        data_out = np.zeros((1,) + out_dim + (1,), dtype=np.float32)
        tmp = data[i * out_dim[0]:i * out_dim[0] + out_dim[0]]
        data_out[0, :tmp.shape[0], ..., 0] = tmp
        preds[i] = model.predict(data_out)

    label_nda = np.reshape(preds, (np.prod(preds.shape[:2]),) + preds.shape[2:])[:data.shape[0]]
    label_nda = label_nda[..., 1]

    print(np.unique(label_nda))

    print("Binarizing to produce label volume")
    th = 0.5
    label_nda[label_nda < th] = 0
    label_nda[label_nda >= th] = 1
    label_nda = label_nda.astype(np.uint8)

    '''
    print(np.unique(label_nda))

    label = sitk.GetImageFromArray(label_nda)
    itkimage = sitk.Cast(sitk.RescaleIntensity(itkimage), sitk.sitkUInt8)
    label.CopyInformation(itkimage)
    '''

    return label_nda, data


def predict_old(input_volume_path, output_mask_path):

    print("Extracting data...")
    itkimage = sitk.ReadImage(input_volume_path)
    data = sitk.GetArrayFromImage(itkimage)

    print(np.unique(data))
    
    print("Pre-processing...")
    # fix orientation
    data = np.rot90(data, k=2, axes=(1, 2))
    data = np.flip(data, axis=2)

    # HU-clip
    limits = (0, 140)
    data[data < limits[0]] = limits[0]
    data[data > limits[1]] = limits[1]

    # scale -> [0, 1]
    data = data - np.amin(data)
    data = data / np.amax(data)

    print("Loading model...")
    # load trained model
    model = load_model(model_path, compile=False)

    print("Predicting...")

    # split data into chunks and predict
    out_dim = (16, 512, 512)
    preds = np.zeros((int(np.ceil(data.shape[0] / out_dim[0])),) + out_dim + (2,)).astype(np.float32)
    for i in range(int(np.ceil(data.shape[0] / out_dim[0]))):
        data_out = np.zeros((1,) + out_dim + (1,), dtype=np.float32)
        tmp = data[i * out_dim[0]:i * out_dim[0] + out_dim[0]]
        data_out[0, :tmp.shape[0], ...,0] = tmp
        preds[i] = model.predict(data_out)

    label_nda = np.reshape(preds, (np.prod(preds.shape[:2]),) + preds.shape[2:])[:data.shape[0]]
    label_nda = label_nda[..., 1]

    print(np.unique(label_nda))

    print("Binarizing to produce label volume")
    th = 0.3
    label_nda[label_nda < th] = 0
    label_nda[label_nda >= th] = 1
    label_nda = label_nda.astype(np.uint8)

    print(np.unique(label_nda))

    label = sitk.GetImageFromArray(label_nda)
    itkimage = sitk.Cast(sitk.RescaleIntensity(itkimage), sitk.sitkUInt8)
    label.CopyInformation(itkimage)

    return label_nda, data


def brainseg(data_orig):
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
    tmp = remove_small_objects(tmp.astype(bool), min_size=0.1 * np.prod(tmp.shape)).astype(int)
    tmps = tmp.copy()
    for i in range(tmps.shape[0]):
        tmps[i] = remove_small_holes(tmps[i].astype(bool), area_threshold=0.1 * np.prod(tmps[i].shape)).astype(int)
        tmps[i] = cv2.dilate(tmps[i].astype(np.uint8), disk(7))
        tmps[i] = cv2.erode(tmps[i].astype(np.uint8), disk(7))
        #tmps[i] = remove_small_holes(tmps[i].astype(bool), area_threshold=0.1 * np.prod(tmps[i].shape)).astype(int)
        tmps[i] = binary_fill_holes(tmps[i])
    # tmps = binary_dilation(tmps, ball(9))
    # tmps = cv2.morphologyEx(tmps.astype(np.uint8), cv2.MORPH_CLOSE, disk(7))
    out_orig[tmps != 1] = 0
    out = out_orig.copy()
    gt = tmps.copy()
    return tmps


if __name__ == "__main__":

    # use single GPU (first one)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # choose set
    sets = "test"

    out_dim = (16, 512, 512)

    # choose patient
    datasets_path = "/home/andrep/workspace/hematoma/output/datasets/"
    name = "16_08_unet_16_512_test_balanced_all_stride_4"  # <--- best?
    name = "14_09_data_16_512_stride_4_balanced"
    name = "19_09_data_16_512_stride_4_balanced_posneg"
    model_name = name

    # load model
    save_model_path = "/home/andrep/workspace/hematoma/output/models/" + "model_" + name + ".h5"
    #save_model_path = "/home/andrep/hematoma/output/models/" + "unet_model.h5"
    model_path = save_model_path
    #model = load_model(save_model_path, compile=False)

    curr = import_set(sets, datasets_path + "dataset_" + name + ".h5")
    uniques = unique_patients(curr)
    shuffle(uniques)

    uniques = [296, 314, 181]  # 275

    for name in uniques:
        #name = uniques[0]
        print('Chosen patient: ')

        #name = "10"
        print(name)

        data_loc = "/home/andrep/workspace/hematoma/data/"
        gt_loc = "/home/andrep/workspace/hematoma/gt/"
        end_path = "/home/andrep/workspace/hematoma/datasets/data_16_512/"
        pred_path = "/home/andrep/workspace/hematoma/output/pred/" + model_name + "/"

        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        '''

        loc = data_loc + str(name) + ".nii.gz"
        gt_loc = gt_loc + str(name) + "/" + str(name) + "-label.nrrd"

        itkimage = sitk.ReadImage(gt_loc)
        gt = sitk.GetArrayFromImage(itkimage)

        # fix orientation
        gt = np.rot90(gt, k=2, axes=(1, 2))
        gt = np.flip(gt, axis=2)

        # get resolution information
        orig_dim = gt.shape
        img_size = out_dim[1:]
        tmp1 = itkimage.GetSpacing()
        tmp2 = itkimage.GetSize()
        res_factor = tmp1[2] / tmp1[0]
        res = np.ones(3)
        res[0] = res_factor
        res[1] = img_size[0] / tmp2[0]
        res[2] = img_size[1] / tmp2[1]

        gt = zoom(gt.astype(np.float32), zoom=res, order=1)
        th = 0.5
        gt[gt <= th] = 0
        gt[gt > th] = 1

        # brain seg
        itkimage = sitk.ReadImage(loc)
        data = sitk.GetArrayFromImage(itkimage)
        data = np.rot90(data, k=2, axes=(1, 2))
        data = np.flip(data, axis=2)
        data_orig = data.copy()
        brain = brainseg(data_orig)
        brain = zoom(brain.astype(np.float32), zoom=res, order=1)
        th = 0.5
        brain[brain <= th] = 0
        brain[brain > th] = 1

        # preprocess and predict
        pred, data = predict(loc, name, out_dim)
        data_orig = data.copy()

        # calculate DSC for each slice
        dsc_list = []
        for i in range(pred.shape[0]):
            dsc_list.append(DSC(pred[i], gt[i]))

        # generate boundary image
        gt_b = np.zeros_like(gt)
        for i in range(gt.shape[0]):
            if len(np.unique(gt[i])) > 1:
                gt_b[i] = cv2.Canny((gt[i] * 255).astype(np.uint8), 0, 255)

        gt_b = gt_b.astype(np.float32)
        gt_b = gt_b / np.amax(gt_b)

        # generate ROI of brain
        brain_b = np.zeros_like(brain)
        for i in range(brain.shape[0]):
            if len(np.unique(brain[i])) > 1:
                brain_b[i] = cv2.Canny((brain[i] * 255).astype(np.uint8), 0, 255)

        brain_b = brain_b.astype(np.float32)
        brain_b = brain_b / np.amax(brain_b)

        f = h5py.File(pred_path + str(name) + '.h5', 'w')
        f.create_dataset('data_orig', data=data_orig)
        f.create_dataset('data', data=data)
        f.create_dataset('gt', data=gt)
        f.create_dataset('gt_b', data=gt_b)
        f.create_dataset('brain', data=brain)
        f.create_dataset('brain_b', data=brain_b)
        f.create_dataset('pred', data=pred)
        f.close()
        
        '''

        f = h5py.File(pred_path + str(name) + '.h5', 'r')
        data_orig = np.array(f["data_orig"])
        data = np.array(f["data"])
        gt = np.array(f["gt"])
        gt_b = np.array(f["gt_b"])
        brain = np.array(f["brain"])
        brain_b = np.array(f["brain_b"])
        pred = np.array(f["pred"])
        f.close()
        #'''

        # calculate DSC for each slice
        dsc_list = []
        for i in range(pred.shape[0]):
            dsc_list.append(DSC(pred[i], gt[i]))



        colors = [(0, 0, 1, i) for i in np.linspace(0, 1, 3)]
        cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
        colors = [(1, 0, 0, i) for i in np.linspace(0, 1, 3)]
        cmap2 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
        colors = [(0, 1, 0, i) for i in np.linspace(0, 1, 3)]
        cmap3 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
        #colors = [(1, 131/255, 0, i) for i in np.linspace(0, 1, 3)]
        colors = [(1, 192/255, 203/255, i) for i in np.linspace(0, 1, 3)]
        cmap4 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
        colors = [(0, 0, 1, i) for i in np.linspace(0, 1, 3)]
        cmap5 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

        f, ax = plt.subplots(1, 3, figsize=(20, 13))
        f.canvas.mpl_connect('key_press_event', up_scroll_alt)
        f.canvas.mpl_connect('key_press_event', down_scroll_alt)
        f.canvas.mpl_connect('scroll_event', up_scroll)
        f.canvas.mpl_connect('scroll_event', down_scroll)

        s1ax = plt.axes([0.25, 0.08, 0.5, 0.03])
        slider1 = Slider(s1ax, 'pred alpha', 0, 1.0, dragging=True, valstep=0.05)

        s2ax = plt.axes([0.25, 0.02, 0.5, 0.03])
        slider2 = Slider(s2ax, 'slice', 0, data.shape[0] - 1, valstep=1, valfmt='%1d')

        s3ax = plt.axes([0.25, 0.20, 0.5, 0.03])
        slider3 = Slider(s3ax, 'threshold', 0, 1.0, dragging=True, valstep=0.05)

        s4ax = plt.axes([0.25, 0.14, 0.5, 0.03])
        slider4 = Slider(s4ax, 'brain alpha', 0, 1.0, dragging=True, valstep=0.05)

        # init
        slider1.set_val(0.3)
        slider2.set_val(0)
        slider3.set_val(0.5) # th = 0.3
        slider4.set_val(0.3)
        f.subplots_adjust(bottom=0.15)

        slider1.on_changed(images)
        slider2.on_changed(images)
        slider3.on_changed(threshold)
        slider3.set_val(slider3.val)
        slider4.on_changed(images)
        #slider2.set_val(slider2.val)


        plt.show()

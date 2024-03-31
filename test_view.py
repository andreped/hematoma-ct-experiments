import matplotlib
matplotlib.use('TkAgg')
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import SimpleITK as sitk
import cv2
from matplotlib.widgets import Slider

def images(event):
	ax.clear()
	ax.imshow(data[int(slider2.val)], cmap = 'gray')
	ax.imshow(gt[int(slider2.val)], cmap=cmap, alpha=float(slider1.val))
	ax.imshow(gt_b[int(slider2.val)], cmap=cmap2)
	ax.set_title('CT + GT')
	ax.set_axis_off()
	
	f.suptitle('slice '+str(int(slider2.val)))
	f.canvas.draw_idle()

def up_scroll_alt(event):
	if event.key == "up":
		if (slider2.val + 2 > data.shape[0]):
			1
			#print("Whoops, end of stack", print(slider2.val))
		else:
			slider2.set_val(slider2.val + 1)
	
def down_scroll_alt(event):
	if event.key == "down":
		if (slider2.val - 1 < 0):
			1
			#print("Whoops, end of stack", print(slider2.val))
		else:
			slider2.set_val(slider2.val - 1)


def up_scroll(event):
	if event.button == 'up':
		if (slider2.val + 2 > data.shape[0]):
			1
			#print("Whoops, end of stack", print(slider2.val))
		else:
			slider2.set_val(slider2.val + 1)


def down_scroll(event):
	if event.button == 'down':
		if (slider2.val - 1 < 0):
			1
			#print("Whoops, end of stack", print(slider2.val))
		else:
			slider2.set_val(slider2.val - 1)



if __name__ == "__main__":

	name = 181

	loc = "/hdd/hematoma/NIFTI/" + str(name) + ".nii.gz"
	gt_loc = "/hdd/hematoma/Hjerneblodninger Emil/" + str(name) + "/" + str(name) + "-label.nrrd"

	itkimage = sitk.ReadImage(loc)
	data = sitk.GetArrayFromImage(itkimage)

	itkimage = sitk.ReadImage(gt_loc)
	gt = sitk.GetArrayFromImage(itkimage)

	print(data.shape)
	print(gt.shape)

	# HU clipping
	limits = (0, 100)
	data[data < limits[0]] = limits[0]
	data[data > limits[1]] = limits[1]

	# fix orientation
	data = np.rot90(data, k=2, axes=(1,2))
	data = np.flip(data, axis=2)
	gt = np.rot90(gt, k=2, axes=(1,2))
	gt = np.flip(gt, axis=2)

	#generate boundary image 
	gt_b = np.zeros_like(gt)
	for i in range(gt.shape[0]):
		if len(np.unique(gt[i])) > 1:
			gt_b[i] = cv2.Canny((gt[i]*255).astype(np.uint8), 0, 255)

	gt_b = gt_b.astype(np.float32)
	gt_b = gt_b / np.amax(gt_b)


	colors = [(0,0,1,i) for i in np.linspace(0,1,3)]
	cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
	colors = [(1,0,0,i) for i in np.linspace(0,1,3)]
	cmap2 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

	f, ax = plt.subplots(1,1, figsize = (12, 12))
	f.canvas.mpl_connect('key_press_event', up_scroll_alt)
	f.canvas.mpl_connect('key_press_event', down_scroll_alt)
	f.canvas.mpl_connect('scroll_event', up_scroll)
	f.canvas.mpl_connect('scroll_event', down_scroll)

	s1ax = plt.axes([0.25, 0.08, 0.5, 0.03])
	slider1 = Slider(s1ax, 'alpha', 0, 1.0, dragging = True, valstep = 0.05)

	s2ax = plt.axes([0.25, 0.02, 0.5, 0.03])
	slider2 = Slider(s2ax, 'slice', 0, data.shape[0]-1, valstep = 1, valfmt = '%1d')

	# init
	slider1.set_val(0.3)
	slider2.set_val(0)
	f.subplots_adjust(bottom = 0.15)

	slider1.on_changed(images)
	slider2.on_changed(images)
	slider2.set_val(slider2.val)

	plt.show()

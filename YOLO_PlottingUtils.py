# John Lambert, Matt Vilim, Konstantine Buhler
# To Train YOLO
# Dec 14, 2016


def plotGroundTruth(annotatedImages):
	"""
	INPUTS:
	-	annotatedImages: Python list of annotated_image class objects
	OUTPUTS:
	-	N/A
	"""
	for imIdx, annotatedImage in enumerate(annotatedImages):
		bboxes = annotatedImage.bounding_boxes
		im = imread( annotatedImage.image_path )
		plotBBoxes(bboxes,im, imIdx)
		if imIdx > 100:
			quit()

def plotBBoxes(bboxes, im, imIdx):
	"""
	INPUTS:
	-	bboxex
	-	im
	-	imIdx
	OUTPUTS:
	-	N/A
	"""
	imWidth = im.shape[1]
	imHeight = im.shape[0]
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.imshow(im, aspect='equal')
	for bbox in bboxes:
		x = bbox.x_min
		y = bbox.y_min
		w = bbox.w
		h = bbox.h
		class_name = bbox.category
		x_cent = x + w/2.
		y_cent = y + h/2.
		x = x_cent
		y = y_cent
		left = max(0, x - w/2. )
		right = min( x + w/2., imWidth-1 )
		top = max(0, y - h/2. )
		bot = min( y + h/2., imHeight-1 )
		ax.add_patch(
	        plt.Rectangle((left, top),
	                      right-left,
	                      bot-top, fill=False,
	                      edgecolor='red', linewidth=3.5)
	        )
		ax.text(left, top-2,
	            '{:s}'.format(class_name ),
	            bbox=dict(facecolor='blue', alpha=0.5),
	            fontsize=14, color='white')
	plt.draw()
	plt.tight_layout()
	plt.savefig( 'Image_%d.png' % imIdx)
	plt.gcf().set_size_inches(15, 12)


def plotGridCellsOnIm(im,ax):
	"""
	Show 7x7 YOLO detection grid on image.
	INPUTS:
	-	im: n-d array image, no specific size required [None x None x 3]
	-	ax: Matplotlib plotting axes
	"""
	for row in range(S):
		for col in range(S):
			imHeight = im.shape[0]
			imWidth = im.shape[1]
			left = row * (1. * imWidth /S )
			right = (row+1) * (1. * imWidth /S)
			top = col * (1. * imHeight/S)
			bot = (col + 1) * (1. * imHeight/S)
			ax.add_patch(
		        plt.Rectangle((left, top),
		                      right-left,
		                      bot-top, fill=False,
		                      edgecolor='red', linewidth=3.5)
		        )
		plt.draw()
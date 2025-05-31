1. General notes:

 You have 2 broad tasks for this project:
 	- Lesion Detection
 	- Lesion Characterization
 
 The lesion data can be found in lesion_data.csv and contains lesions marked by MDs across all images. Both MDs used the Syntax Score methodology for marking and characterizing lesions: https://syntaxscore.org/index.php/tutorial/definitions
 
 For lesion detection, the task can be thought of as approximating a bounding box around areas identified as lesions. You can treat the problem in multiple ways, for example: object detection (fine-tuning a model like YOLO), classification (use a sliding window to take smaller parts of the image and classify whether there's a lesion inside with a vision model (e.g. vision transformer, CNN, etc...)), unsupervised clustering based on local image features, etc...
 
 For lesion characterization, the task is to identify the characteristics of an already-detected lesion, as per Syntax Score. Each of the characteristics can be approached in a multitude of ways e.g. classical (non-ML) methods, SVM over local image features, deep learning etc... My advice would be to first analyze how much each feature is present in the dataset (as some have very few positive examples), then choose a couple to focus on and only do the rest if time allows for it.



2. Files included:

 base_images:
	- base images of the arteries with the following naming format: {image_id}_{frame}.png
	- segmentation_df (csv file containing the following important columns):
		-> image_id 
		-> frame
		-> segmentation
		-> overlap
		-> user
		
		Some segmentations are done on 2 different layers, since they contain overlapping vessels. Both 'segmentation' and 'overlap' are encoded into strings, they can be decoded using the 'unpack_mask' function from utils.py, resulting in a semantic segmentation of the arteries. 2 different MDs labeled the data, hence the same image might have 2 different segmentations (done on the same frame, or different frames) by the 2 doctors.


 graphs:
 	- each pickled file is a dict that contains the following data:
 		-> data = {'graph': graph,	# networkx multigraph
            		'current_image': skeleton,	# centerline obtained from segmentation
            		'segmentation': segmentation, # segmentation (with segmentation and overlap sections merged)
            		'image1': image, # base image
            		'side': side,
            		'primary_angle': angle_1,
            		'secondary_angle': angle_2,
            		'image_id': filename, # image id
            		'frame': frame, # frame
            		'index': index,
            		}
 
		-> Each graph contains the following data inside:
			Nodes: (x, y), Data: {}
			Edge: (x1, y1) <-> (x2, y2), Key: {key}, Data: dict_keys(['length', 'pixels', 'label', 'color'])
			(pixels here is a collection of coordinates of the pixels making up the centerline of each edge)
	
			
 lesion_data:
 	- csv file containing data about all lesions marked by MDs
 	- has the following important columns:
 		-> user (MD who marked the lesion)
 		-> image_id (id of the image on which the lesion was marked)
 		-> frame (frame of the image on which the lesion was marked)
 		-> lesion_x
 		-> lesion_y
 		-> lesion_width
 		-> lesion_height
 		-> lesion_{X} (all other relevant characteristics for Syntax Score marked by MDs)
 		
 
 pixel_spacing: (up to you if you want to try and use this or not)
 	- csv file containing the scale of each image by iamge id (the distance between pixels in mm)
 
 
 utils:
 	- contains:
 		-> unpack_mask: a function for unpacking masks from the csv file in base_images
 		-> COLOR_LABEL_DICT: a dictionary mapping each label from the semantic segmentation to its corresponding Syntax Score label, as well as assigning it a color
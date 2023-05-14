#%%
import cv2
import joel3d as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import pickle
import time
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import os
import glob
import cv2
import numpy as np
#ImageOps
from PIL import Image, ImageOps
from scipy import ndimage
from scipy import misc
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt
import skimage
import cv2
import joel3d as mp
from mediapipe.python.solutions import selfie_segmentation, drawing_utils, face_mesh, face_detection
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA

#crop and align face
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

im_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\mediapipe-facemesh\data\gakki.jpg"


image = cv2.imread(im_path)

def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255))
    lower_threshold = np.array([0, 15, 0], dtype=np.uint8)
    upper_threshold = np.array([17,170,255], dtype=np.uint8)
    # Defining HSV Threadholds
    # lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    # upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=8, hasThresholding=True):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img, sample_weight=None)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


"""## Section Two.4.2 : Putting it All together: Pretty Print
The function makes print out the color information in a readable manner
"""


def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()
def get_mesh_contours_annotation_image_landmarks(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    mp_face_detection = mp.solutions.face_detection

    drawing_spec = mp_drawing.DrawingSpec(thickness=25, circle_radius=0.1)
    drawing_spec2 = mp_drawing.DrawingSpec(thickness=1, circle_radius=10)
    contours = np.zeros_like(image)
    mesh = np.zeros_like(image)
    annotated_image = np.empty_like(image)
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:
        image_height, image_width, _ = image.shape
        results = holistic.process(image)        
        if results.face_landmarks:
            annotated_image = image.copy()
            contours = np.zeros_like(image)
            contours2 = np.zeros_like(image)
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (0, 0, 0)
            annotated_image = np.where(condition, annotated_image, bg_image)
            mp_drawing.draw_landmarks(
                contours,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(
                mesh,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec2)
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec2)

        return mesh, contours, annotated_image, image, results.face_landmarks.landmark
def crop(image):
    #get face detection
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.3) as face_detection:
        results = face_detection.process(image)
        if results.detections:
            for detection in results.detections:
                #set bounding box size 
                box = detection.location_data.relative_bounding_box
                # center = detection.location_data.relative_keypoints[0]
                print("box = {box}")
                print(f"image height: {image.shape[0]} image width: {image.shape[1]} image shape: {image.shape}")
                x = box.xmin 
                x = int(x * image.shape[1])
                y = box.ymin
                y = int(y * image.shape[0])
                w = box.width
                w = int(w * image.shape[1])
                h = box.height
                h = int(h * image.shape[0])
                print(x,y,w,h)
                #make the bounding box larger
                # x = x - 50
                # y = y - 200
                # w = w + 100
                # h = h + 250
                try :
                    face = image[y:y+h, x:x+w]
                except:
                    print("error")
                    face = image
                return face

def remove_background(image):
    # Load the face detection calculator graph.
    face_detection_calculator = face_detection.FaceDetection()
    # Run the face detection calculator.
    faces = face_detection_calculator.process(image)

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
    #show face only
    f = selfie_segmentation.process(image)
    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # get the result
    results = selfie_segmentation.process(RGB)
    # extract segmented mask
    mask = results.segmentation_mask
    # apply mask to the original image
    mask = np.stack((mask,)*3, axis=-1)
    mask = mask * 255
    mask = mask.astype(np.uint8)
    # apply mask to the original image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    
def center_align_crop(image):
    
    landmarks = get_mesh_contours_annotation_image_landmarks(image)[4]
    #get left eye and right eye
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    #get angle between eyes
    angle = np.arctan2(left_eye.y - right_eye.y, left_eye.x - right_eye.x)
    # angle = -(angle * 180) / np.pi

    #rotate image by angle
    image = ndimage.rotate(image, -angle)

    # print(left_eye)
    # print(right_eye)
    return image

        
if __name__ == "__main__":

    no_background = remove_background(image)

    aligned = center_align_crop(no_background)

    mesh, contours, annotated_image, image, land_marks = get_mesh_contours_annotation_image_landmarks(aligned)
    plt.tight_layout()

    #plot all 4 images side by side
    fig, axs = plt.subplots(1, 4, figsize=(20, 20))
    axs[0].imshow(image)
    axs[0].set_title("image")
    axs[1].imshow(contours)
    axs[1].set_title("contours")
    axs[2].imshow(mesh)
    axs[2].set_title("mesh")
    axs[3].imshow(annotated_image)
    axs[3].set_title("annotated_image")
    plt.title("before cropping")
    plt.show()

    cropped_image = crop(image)
    cropped_mesh, cropped_contours, cropped_annotated_image, cropped_image, lm = get_mesh_contours_annotation_image_landmarks(cropped_image)
    #plot all 4 images side by side
    fig, axs = plt.subplots(1, 4, figsize=(20, 20))
    axs[0].imshow(cropped_image)
    axs[0].set_title("image")
    axs[1].imshow(cropped_contours)
    axs[1].set_title("contours")
    axs[2].imshow(cropped_mesh)
    axs[2].set_title("mesh")
    axs[3].imshow(cropped_annotated_image)
    axs[3].set_title("annotated_image")
    plt.title("after cropping")
    plt.show()

    #extract dominant color and plot it
    # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
    dominantColors = extractDominantColor(cropped_image, hasThresholding=True)


    fig, axs = plt.subplots(2, 1, figsize=(30, 20))
    # Show in the dominant color as bar

    colour_bar = plotColorBar(dominantColors)
    #convert to rgb
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    axs[0].imshow(cropped_image)
    axs[1].imshow(colour_bar)
    plt.show()
# %%

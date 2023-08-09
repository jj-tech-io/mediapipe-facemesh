from scipy.spatial import Delaunay
import cv2
import numpy as np
import mediapipe as mp
import skimage
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import matplotlib.pyplot as plt

def get_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i in range(0, 468):
                landmark = face_landmarks.landmark[i]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                landmarks.append((x, y))
    return np.array(landmarks)

def warp_image(image1, image2, landmarks1, landmarks2):
    # Compute Delaunay Triangulation
    delaunay = Delaunay(landmarks1)
    warped_image = np.zeros_like(image1)
    
    # Iterate through each triangle in the triangulation
    for simplex in delaunay.simplices:
        # Get the vertices of the triangle in both images
        src_triangle = landmarks1[simplex]
        dest_triangle = landmarks2[simplex]

        # Compute the bounding box of the triangle in both images
        src_rect = cv2.boundingRect(np.float32([src_triangle]))
        dest_rect = cv2.boundingRect(np.float32([dest_triangle]))

        # Crop the triangle from the source and destination images
        src_cropped_triangle = image1[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
        dest_cropped_triangle = np.zeros((dest_rect[3], dest_rect[2], 3), dtype=np.float32)

        # Adjust coordinates to the cropped region
        src_triangle_adjusted = src_triangle - (src_rect[0], src_rect[1])
        dest_triangle_adjusted = dest_triangle - (dest_rect[0], dest_rect[1])

        # Compute the affine transformation
        matrix = cv2.getAffineTransform(np.float32(src_triangle_adjusted), np.float32(dest_triangle_adjusted))

        # Warp the source triangle to the shape of the destination triangle
        warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (dest_rect[2], dest_rect[3]))

        # Mask for the destination triangle
        mask = np.zeros((dest_rect[3], dest_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dest_triangle_adjusted), (1, 1, 1), 16, 0)

        # Place the warped triangle in the destination image
        warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = \
            warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] * (1 - mask[:, :, None]) \
            + warped_triangle * mask[:, :, None]

    return warped_image.astype(np.uint8)

image1 = cv2.imread(r"1_neutral.jpg")
image2 = cv2.imread(r"images/models_4k/m32.png")
WIDTH = 2048
HEIGHT = 2048
image2 = cv2.resize(image2, (WIDTH, HEIGHT))
image1 = cv2.resize(image1, (WIDTH, HEIGHT))

landmarks1 = get_landmarks(image1)
landmarks2 = get_landmarks(image2)

# Warp image1 to align with image2
warped_image1 = warp_image(image1, image2, landmarks1, landmarks2)

warped_landmarks1 = get_landmarks(warped_image1)

plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title("Source image")
plt.show()
plt.imshow(cv2.cvtColor(warped_image1, cv2.COLOR_BGR2RGB))
plt.title("Warped image")
plt.show()

plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title("Target image")
plt.show()

#plot 1 on top of 2 with alpha 0.5
alpha = 0.5
beta = (1.0 - alpha)
dst = cv2.addWeighted(warped_image1, alpha, image2, beta, 0.0)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title("Overlay")
plt.show()

unwarped_image = warp_image(warped_image1, image1, warped_landmarks1, landmarks1)
plt.imshow(cv2.cvtColor(unwarped_image, cv2.COLOR_BGR2RGB))
plt.title("Unwarped image")
plt.show()
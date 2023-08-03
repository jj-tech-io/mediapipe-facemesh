#%
import numpy as np
import matplotlib.pyplot as plt
import cv2
""" 
# Image size
size = (512, 512)

# Create an empty image
image = np.zeros(size)

# Center of the blemish
cx, cy = size[0] // 2, size[1] // 2

# Standard deviation of the Gaussian
sigma = 50

# Intensity of the blemish
intensity = 1.0

# Create the blemish
for x in range(size[0]):
    for y in range(size[1]):
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        image[x, y] += intensity * np.exp(-(dist**2) / (2 * sigma**2))

# Show the image
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.show()
"""
Cm_path = r"images/m141_Cm.png"
Ch_path = r"images/m141_Ch.png"

#load Cm and Ch as binary images
Cm = plt.imread(Cm_path).astype(np.float32)
Ch = plt.imread(Ch_path).astype(np.float32)
#resize to 512x512
Cm = cv2.resize(Cm, (512, 512))
Ch = cv2.resize(Ch, (512, 512))

def gaussian(image, cx, cy, sigma, intensity):
    modified_image = image.copy()
    rows, cols, channels = modified_image.shape
    for x in range(rows):
        for y in range(cols):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            modified_image[x, y] += intensity * np.exp(-(dist**2) / (2 * sigma**2))
            # Clip the values to make sure they remain within the valid range
            modified_image[x, y] = np.clip(modified_image[x, y], 0, 1)
    return modified_image


gaus = gaussian(Cm, 250, 150, 2, 1.0)

#%
plt.imshow(gaus, cmap='binary', vmin=0, vmax=1)
plt.show()
# %

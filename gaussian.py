import numpy as np
import matplotlib.pyplot as plt

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

#create function to generate gaussian
def gaussian(size, sigma, intensity):
    # Create an empty image
    image = np.zeros(size)

    # Center of the blemish
    cx, cy = size[0] // 2, size[1] // 2

    # Create the blemish
    for x in range(size[0]):
        for y in range(size[1]):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            image[x, y] += intensity * np.exp(-(dist**2) / (2 * sigma**2))
    return image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def overlay_images(background, overlay, alpha, x=0, y=0):
    # Ensure the images have the same shape
    assert background.shape == overlay.shape

    # Convert uint8 to float
    background = background.astype(float)
    overlay = overlay.astype(float)

    # Image ranges from 0 to 255
    background /= 255
    overlay /= 255

    # Compute the weighted average
    background = background * (1 - alpha)
    overlay = overlay * alpha
    output = background + overlay

    # Convert back to uint8
    output *= 255
    output = output.astype(np.uint8)

    return output

# Load grayscale background and overlay images
background_image = np.array(Image.open("test0.png"))
overlay_image = np.array(Image.open("test1.png"))


# Set the alpha value (transparency level)
alpha_value = 0.05

# Overlay the images
result_image = overlay_images(background_image, overlay_image, alpha_value)

# Display the result
plt.imshow(result_image, cmap='gray')  # Use 'gray' colormap for grayscale images
plt.axis('off')
plt.show()
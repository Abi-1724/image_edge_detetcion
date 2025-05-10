import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Grayscale Image", image)

# Convert to array of numbers
image_array = np.array(image)
print("Image as array:\n", image_array)

# Apply Sobel edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Frequency domain filtering - FFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Create a low-pass filter mask
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros_like(image, dtype=np.uint8)
r = 30  # radius
cv2.circle(mask, (ccol, crow), r, 255, -1)
fshift_filtered = fshift * (mask / 255)

# Inverse FFT to get filtered image
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Show results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Edge Detection (Sobel)")
plt.imshow(sobel_combined, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Low-Pass Filtered Image")
plt.imshow(img_back, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("FFT Spectrum")
plt.imshow(np.log(1 + np.abs(fshift)), cmap='gray')

plt.tight_layout()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

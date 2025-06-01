import cv2
import numpy as np

def apply_skew_correction(image):
    """
    Correct skew in the image using edge detection and affine transformation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use HoughLines to find the most dominant line (which should align with the plate)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for rho, theta in lines[0]:
            angle = (theta - np.pi / 2) * 180 / np.pi  # Calculate angle relative to vertical
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image



def apply_thresholding(image):
    """
    Apply adaptive thresholding to binarize the image.
    Converts the image to grayscale if it's not already in that format.
    """
    # Check if the image is already grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        # Image is already grayscale, no need to convert
        pass
    else:
        raise ValueError("Unsupported image format. Image must be either a 2D grayscale or 3-channel BGR image.")

    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
    return thresholded




img = cv2.imread("../4ca505b8-b568-42bb-bc2f-939cf71d3e58-1719540047.jpg")
img_2 = apply_thresholding(img)

cv2.imshow("Prediction", img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()



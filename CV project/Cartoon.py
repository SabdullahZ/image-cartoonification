import cv2
import numpy as np

def read_img(filename):
    img = cv2.imread(filename)
    if img is None:
        raise Exception(f"Error loading image from {filename}. Check the file path.")
    return img

def edge_detection(img, line_wdt, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayblur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(grayblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_wdt, blur)
    return edges

def color_quantisation(img, totalColors):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, totalColors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

def smooth(img, img_dir):
    width = img.shape[1]
    height = img.shape[0]
    dim = (200, 200)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # smooth image
    smoothed = resized
    for _ in range(5):
        smoothed = cv2.bilateralFilter(smoothed, 9, 75, 75)
    # resize back
    dim = (width, height)
    smoothed = cv2.resize(smoothed, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(img_dir + "smoothed.jpg", smoothed)

    return smoothed

try:
    filename = './save.jpeg'
    img = read_img(filename)
    line_wdt = 9
    blur_value = 7
    totalColors = 14

    edgeImg = edge_detection(img, line_wdt, blur_value)
    img = color_quantisation(img, totalColors)
    blurred_img = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)

    edgeImg = cv2.cvtColor(edgeImg, cv2.COLOR_GRAY2BGR)
    edgeImg = cv2.resize(edgeImg, (blurred_img.shape[1], blurred_img.shape[0]))

    # Apply smoothing function
    smoothed_img = smooth(blurred_img, './')  # Adjust the directory as needed

    # Convert the images to 8-bit unsigned integer (CV_8U) data type
    blurred_img = cv2.convertScaleAbs(blurred_img)
    edgeImg = cv2.convertScaleAbs(edgeImg)

    # Ensure the mask (edgeImg) is in grayscale and binary format
    edgeImg = cv2.cvtColor(edgeImg, cv2.COLOR_BGR2GRAY)
    _, edgeImg = cv2.threshold(edgeImg, 1, 255, cv2.THRESH_BINARY)

    # Perform bitwise AND operation
    cartoon = cv2.bitwise_and(smoothed_img, smoothed_img, mask=edgeImg)

    # Resize the cartoon image for display
    scaled_cartoon = cv2.resize(cartoon, (500, 600))  # Adjust the size as needed

    # Display the resized cartoon image
    cv2.imshow('Cartoon Image', scaled_cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")

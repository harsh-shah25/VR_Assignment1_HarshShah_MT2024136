#using WATERSHED ALGO

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_coins(image_path):

    image = cv2.imread(image_path)
    og_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 30, 60)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Changing white pixels to green
    edges_colored[np.where((edges_colored == [255, 255, 255]).all(axis=2))] = [0, 255, 0]

    overlay = cv2.addWeighted(image, 1, edges_colored, 1, 0)

    # Thresholding
    ret, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY) 
    ret, thresh2 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV) 
    
    contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    im2 = np.zeros_like(gray)  
    cv2.drawContours(im2, filtered_contours, -1, (255, 255, 255), cv2.FILLED) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    sure_bg = cv2.dilate(im2, kernel, iterations=3)

    dist = cv2.distanceTransform(im2, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)

    sure_fg = sure_fg.astype(np.uint8)

    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers+1
    
    unknown = cv2.subtract(sure_bg,sure_fg)
    markers[unknown==255] = 0

    markers = cv2.watershed(image,markers)
    image[markers == -1] = [255,0,0]
    total_coins = ret-1
    print(f"Total number of coins in the image : {total_coins}")
    
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

    plt.subplot(2,3,2)
    plt.title("Visualized edges")
    plt.imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))

    plt.subplot(2,3,3)
    plt.title("inverted_binary")
    plt.imshow(thresh2,cmap="gray")


    plt.subplot(2,3,4)
    plt.title("Detected contours")
    plt.imshow(im2)

    plt.subplot(2,3,5)
    plt.title("Sure foreground")
    plt.imshow(sure_fg)
    
    plt.subplot(2,3,6)
    plt.title("Marker Image")
    plt.imshow(markers)

    plt.show()

# Example usage
detect_coins('coins/coins8.jpeg')

# VR_Assignment1_HarshShah_MT2024136
## Coin Detection and Image Stitching to form Panorama
### Coin Detection

**Edge Detection**\
The image of coins was converted to grayscale and then smoothened using a *Gaussian filter* and the *Canny edge detection* was used to detect the edges in the image. The detected edges were overlayed on the original image. 

**Thresholding**\
*Inverse Binary thresholding* was used to make the image binary with black in the background. Due to the designs on the coin there were some black patches inside the coin(white area) and they were not vanishing after morphological transformations, so *Contour detection* was used to eliminate them completely.

**Contours detection and drawing**\
*Contour detection* was used to surround the coins and fill(cv2.FILLED) the gaps inside coins. Again, the noisy contours were rejected by a simple if condition checking if the area is greater than 500, giving us the *filtered_contours*. The final count of coins was equal to the number of filtered_contours. But this much processing is not enough for touching coins because we get only 1 contour in this so we will use watershed algorithm.

**Watershed Algorithm**\
This algorithm was used to select the sure foreground(the area which surely represents the coins) lastly cv2.connectedComponents() was used to count the number of connected white pixels. Each connected component represents a coin. So the number of coins is equal to the number of components. The cv2.watershed() algorithm gives unique markers to each components. 

**Normal Image of Coins:**
![image](https://github.com/user-attachments/assets/f733c97c-5554-4bff-8cc4-1725609513ea)

**Image of touching Coins:**
![image](https://github.com/user-attachments/assets/d7d4f944-0cbb-4ed1-a9e2-f42f8b2e5ad7)


### Image Stitching to form Panorama



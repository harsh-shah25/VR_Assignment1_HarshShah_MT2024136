# VR_Assignment1_HarshShah_MT2024136
## Coin Detection and Image stitching to form panorama
### Coin Detection

**Edge Detection**
The image of coins was converted to grayscale and then smoothened using a **Gaussian filter** and the **Canny edge detection** was used to detect the edges in the image. The detected edges were overlayed on the original image. 

**Thresholding**
**Inverse Binary thresholding** was used to make the image binary with black in the background. Due to the designs on the coin there were some black patches inside the coin(white area) and they were not vanishing after morphological transformations, so **Contour detection** was used to eliminate them completely.

**Contours detection and drawing**
**Contour detection** was used to fill the gaps inside coins. 

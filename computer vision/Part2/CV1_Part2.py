from email.encoders import encode_base64
from pickletools import uint8
from traceback import print_tb
from unittest import findTestCases
import numpy as np
import cv2

def median_filter(img,size=3):
    rows,cols = img.shape
    filtered = np.zeros((rows,cols),np.uint8)
    
    for y in range(size//2,rows-(size//2)+1):
        for x in range(size//2,cols-(size//2)):
            median_arr = img[(y-size//2):(y+((size//2))+1),(x-size//2):(x+((size//2))+1)]
            median_vector = median_arr.flatten()
            median_vector = sorted(median_vector)
            filtered[y,x] = median_vector[len(median_vector)//2]
            
    return filtered

def Gaussian_smoothing(img,sigma=1,size=3):
    kernel = np.zeros((size,size), np.float32)
    origin = size//2
    for y in range(size):
        for x in range(size):
            eclidian_distance = np.sqrt((y-origin)**2 + (x-origin)**2)
            kernel[y,x] = np.exp(-(eclidian_distance**2/(2*(sigma**2))))
    kernel /= np.sum(kernel)

    rows,cols = img.shape
    smoothed = np.zeros((rows,cols),np.uint8)

    for y in range(size//2,rows-(size//2)):
        for x in range(size//2,cols-(size//2)):
            guassian_arr = img[(y-size//2):(y+((size//2)+1)),(x-size//2):(x+((size//2)+1))]
            mul = guassian_arr*kernel
            new_pixel = sum(map(sum,mul))
            smoothed[y,x] = new_pixel
    return smoothed
    
def sobel_filter(img,sigma=1,size=3):
    rows,cols = img.shape
    horizontal_gradiant = np.zeros((rows,cols),np.float32)
    vertical_gradiant = np.zeros((rows,cols),np.float32)
    horizontal_gradiant_f = np.zeros((rows,cols),np.float32)
    vertical_gradiant_f = np.zeros((rows,cols),np.float32)
    kernel = np.zeros((size,size), np.float32)
    origin = size//2
    for y in range(size):
        for x in range(size):
            eclidian_distance = np.sqrt((y-origin)**2 + (x-origin)**2)
            kernel[y,x] = np.exp(-(eclidian_distance**2/(2*(sigma**2))))
    kernel /= np.sum(kernel)
    gaussian_hoz = np.reshape(kernel[0,:],(-1,1))
    gaussian_ver = np.reshape(kernel[:,0],(1,-1))
    sobel = [val for val in range(size//2,-size//2,-1)]
    sobel_hoz = np.reshape(sobel,(1,-1))
    sobel_ver = np.reshape(sobel,(-1,1))
    print(gaussian_hoz)
    print(gaussian_ver)
    print(sobel_hoz)
    print(sobel_ver)

    for y in range(size//2, rows-size//2):
        for x in range(size//2, cols - size//2):
            ver_temp = img[y-size//2:y+size//2+1,x]
            ver_temp = np.reshape(ver_temp,(-1,1))

            #getting horizontal gradiant
            hoz_mul = ver_temp*gaussian_hoz
            hoz_mul = hoz_mul.flatten()
            hoz_grad = sum(hoz_mul)


            #getting vertical gradiant
            ver_mul = ver_temp*sobel_ver
            ver_mul = ver_mul.flatten()
            ver_grad = sum(ver_mul)
            

            horizontal_gradiant[y,x] = hoz_grad
            vertical_gradiant[y,x] = ver_grad
    for y in range(size//2, rows-size//2):
        for x in range(size//2, cols-size//2):
            # hoz_temp = gradiant[y,x-size//2:x+size//2+1]
            hoz_temp = horizontal_gradiant[y, (x - size//2):(x + size//2 + 1)]
            hoz_temp1 = vertical_gradiant[y, (x - size//2):(x + size//2 + 1)]
            
           
            #getting horizontal gradiant
            hoz_mul = hoz_temp*sobel_hoz
            hoz_mul = hoz_mul.flatten()
            hoz_grad = sum(hoz_mul)

            #getting vertical gradiant
            ver_mul = hoz_temp1*gaussian_ver
            ver_mul = ver_mul.flatten()
            ver_grad = sum(ver_mul)


            horizontal_gradiant_f[y,x] = hoz_grad
            vertical_gradiant_f[y,x] = ver_grad
    
    return horizontal_gradiant_f,vertical_gradiant_f

        

def non_max_suppression(horiz_grad,vert_grad,img,size=3):
    rows,cols = horiz_grad.shape
    d = np.zeros((rows,cols),np.float32)
    suppressed = np.zeros((rows,cols),np.float32)
    for y in range(rows):
        for x in range(cols):
            q=255
            r=255
            d[y,x] = np.arctan2(vert_grad[y,x],horiz_grad[y,x])*180/np.pi
    for y in range(size//2,rows-size//2):
        for x in range(size//2,cols-size//2):
            if (0 <= d[y,x] < 22.5) or (157.5 <= d[y,x] <= 180):
                q = img[y, x+1]
                r = img[y, x-1]
            #angle 45
            elif (22.5 <= d[y,x] < 67.5):
                q = img[y+1, x-1]
                r = img[y-1, x+1]
            #angle 90
            elif (67.5 <= d[y,x] < 112.5):
                q = img[y+1, x]
                r = img[y-1, x]
            #angle 135
            elif (112.5 <= d[y,x] < 157.5):
                q = img[y-1, x-1]
                r = img[y+1, x+1]
            if (img[y,x] >= q) and (img[y,x] >= r):
                suppressed[y,x] = img[y,x]
            else:
                suppressed[y,x] = 0
    return suppressed

def thresh(edge_map,weak_thresh,strong_thresh,strong_val=255,weak_val=150):
    rows,cols = edge_map.shape
    threshed = np.zeros((rows,cols),dtype=np.uint8)
    canny_res = np.zeros((rows,cols),dtype=np.uint8)
    for y in range(1,rows-1):
        for x in range(1,cols-1):
            if edge_map[y,x] >= strong_thresh:
                threshed[y,x]=strong_val
            elif edge_map[y,x] < strong_thresh and edge_map[y,x] >= weak_thresh:
                threshed[y,x]=weak_val
            else:
                threshed[y,x]=0
    for y in range(1,rows-1):
        for x in range(1,cols-1):
            if threshed[y,x] == weak_val:
                if (threshed[y-1,x-1] == strong_thresh or threshed[y-1,x] == strong_thresh or threshed[y-1,x+1] == strong_thresh or
                    threshed[y,x-1] == strong_thresh                      or                  threshed[y,x+1] == strong_thresh or
                    threshed[y+1,x-1] == strong_thresh or threshed[y+1,x] == strong_thresh or threshed[y+1,x+1] == strong_thresh):
                        threshed[y,x] = strong_val
                else:
                    threshed[y,x] = 0


    return threshed

def hough_lines(edge_map):
    rows,cols = edge_map.shape
    max_rho = int(round(np.sqrt(rows**2+cols**2)))
    hough_space = np.zeros((max_rho,180))
    for y in range(rows):
        for x in range(cols):
            if edge_map[y,x]==255:
                for theta in range(180):
                    rho = x*np.cos(theta) + y*np.sin(theta)
                    hough_space[int(rho),theta] += 1
    return hough_space

def extract_lines(hough_space,votes):
    line_param = []
    rows,cols = hough_space.shape
    for rho in range(rows):
        for theta in range(cols):
            if hough_space[rho,theta] >= votes:
                line_param.append([rho,theta])
    return line_param
                    



# total functions: 7
# median_filter: returns filtered image using median filter
# Gaussian_smoothing(optional): returns filtered image using gaussina filter
# sobel_filter: generate separable sobel filter and returns horizontal and vertical gradiants
# non_max_suppression: returns suppressed version of the edge map
# thresh: return result of hysteresis thresholding
# hough_lines: returns hough space with votes for every pixel
# extract_lines: returns list of rhos and thetas that meet given votes





##  Loading image ##
img = cv2.imread("3.jpg",cv2.IMREAD_GRAYSCALE)
####################


## Applying median filter ##
# parameters:
#           image,(default filter size=3)

#2.png filter sizes(uncomment when needed)

# filtered = median_filter(img.copy(),size=3)
# horizontal_gradiant,vertical_gradiant = sobel_filter(filtered,size=5)

#3.jpg filter sizes

filtered = median_filter(img.copy(),size=7)
#############################

##              Canny               ##
 
## Applying sobel filter ##
# parameters:
#           image(output of median filter),(default filter size=3)
horizontal_gradiant,vertical_gradiant = sobel_filter(filtered,size=5)
hoz = horizontal_gradiant**2;
ver = vertical_gradiant**2;
edge = np.sqrt((ver+hoz),dtype=np.float32)*16 #getting the gradiant magnitude for the edge map
###########################


## Adding non-max Suppression and Hysteresis Thresholding ##
# functions:
#           1)non_max_suppression(horizontal gradiant, vertical gradiant, edge map)
#           2)thresh(suppressed image, weak pixels,strong pixels,<default strong value=255,<default weak value=150>)
suppressed = non_max_suppression(horizontal_gradiant,vertical_gradiant,edge)
threshed = thresh(suppressed,70,150)

roi = (threshed.shape[0]//2)+100 # making regoin of interset in the y axis
threshed[:roi,:] = 0 # every pixel from the beginning till the roi row equal 0

# adjusting numpy array data type for opencv image show function
edge = np.array(edge,np.uint8)
suppressed = np.array(suppressed,np.uint8)
#############################################################


##############################################


## applying Hough transform ##
# functions:
#           1)hough_lines(filtered edge map with values = 255, 150 or 0)           
#           2)extract_lines(hough space returned form hough_lines, votes for every line)
hough_space = hough_lines(threshed)

#2.png votes
# lines = extract_lines(hough_space,75)

lines = extract_lines(hough_space,170)

line_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for line in lines:
    rho,theta = line
    a = rho*np.cos(theta)
    b = rho*np.sin(theta)
    p1 = (int(a + 1000*(-b)), int(b + 1000*(a)))
    p2 = (int(a - 1000*(-b)), int(b - 1000*(a)))
    if p2[0]-p1[0]==0:
        continue
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p1[1]-m*p1[0]

    y1 = img.shape[0]
    y2 = roi 

    x1 = int((y1-b)/m)
    x2 = int((y2-b)/m)

    cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),1,cv2.LINE_AA)
#################################



## Showing results ##
compare = np.concatenate((img,filtered,edge,suppressed,threshed),axis=1)
cv2.imwrite('compare.jpeg',compare)
cv2.imwrite('cannyFiltered.jpeg',threshed)
cv2.imshow('filtered', compare)
cv2.imshow('output',line_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
####################


import numpy as np
import cv2
import os
# Path to load the model 
PROTOTXT = r"C:\Users\Fatima\img-coloring (DNN)\model\colorization_deploy_v2 (1).prototxt"
POINTS = r"C:\Users\Fatima\img-coloring (DNN)\model\pts_in_hull (1).npy"
MODEL = r"C:\Users\Fatima\img-coloring (DNN)\model\colorization_release_v2 (1).caffemodel"
img_path =r"C:\Users\Fatima\img-coloring (DNN)\BW-Colored\na5lat.jpg"


# loadung the colorization model from Caffe using the prototxt and model weight files
net = cv2.dnn.readNetFromCaffe(PROTOTXT,MODEL)
#Loads the cluster points (313 color bins) from the .npy file.
points = np.load(POINTS)


#cheacking layer information
for i in range(net.getLayerId("class8_ab") + 1):
    try:
        layer = net.getLayer(i)
        print(f"Layer {i}: {layer.name}")
    except:
        pass


#points are transposed and reshaped to fit the model's expected format
points = points.transpose().reshape(2,313,1,1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313],2.606,dtype="float32")]




bw_image = cv2.imread(img_path) #loading the image
normalized = bw_image.astype("float32") / 255.0 #normalization
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB) #convert bgr-->lab
resized = cv2.resize(lab,(244,244)) #rsize 244*244 reqyired by the model
L = cv2.split(resized)[0] #extact l chanel (gray part)
L -=50

#feeding the image to the network
net.setInput(cv2.dnn.blobFromImage(L))

#predecting the colores
ab = net.forward()[0, :, :, :].transpose((1,2,0))

ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0])) #back to the original size 
L = cv2.split(lab)[0] #extract the L channel from the original img

colorized = np.concatenate((L[:,:,np.newaxis], ab), axis = 2) # combine the original l with predected ab
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR) #convert from the lab to bgr 
colorized = (255.0 * colorized).astype("uint8")

#displayin the image
max_dim = 800

def resize_for_display(img, max_dim):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# Generate output filename (using original image name)
output_dir = r"C:\Users\Fatima\img-coloring (DNN)\BW-Colored" 
output_filename = os.path.join(output_dir, "colorized_" + os.path.basename(img_path))
cv2.imwrite(output_filename, colorized)
print(f"Colorized image saved to: {output_filename}") 


display_bw = resize_for_display(bw_image, max_dim)
display_color = resize_for_display(colorized, max_dim)
cv2.imshow("BW img", display_bw)
cv2.imshow("Colorized", display_color)
cv2.imwrite("colorized_output.png", colorized) #saving the results

cv2.waitKey(0)
cv2.destroyAllWindows()

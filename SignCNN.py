#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers.core import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers
import cv2


# In[2]:


def get_image_size():
    img = cv2.imread('C:/Users/Tanmay/Desktop/HandGesture/digit/Train/1/1.jpg', 0)
    print( img.shape)
    return img.shape


# In[20]:


classifier = Sequential()
#classifier.add(ZeroPadding2D((2, 2), input_shape=(64,64,3)))   
image_x, image_y = get_image_size()

classifier.add(Convolution2D(32,(2,2),input_shape = (image_x,image_y,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2), strides=(2, 2), padding='same'))



#classifier.add(ZeroPadding2D((2, 2)))
classifier.add(Convolution2D(64,(2,2), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2), strides=(2, 2), padding='same'))

#classifier.add(ZeroPadding2D((2, 2)))
"""
classifier.add(Convolution2D(128,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3,3), strides=(3, 3), padding='same'))

classifier.add(ZeroPadding2D((2, 2)))
classifier.add(Convolution2D(128,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(128,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
"""
#classifier.add(Dropout(0.2))
classifier.add(Flatten())

#classifier.add(Dense(output_dim = 256, activation = 'tanh'))
#classifier.add(Dense(output_dim = 256, activation = 'tanh'))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dropout(0.2))
"""classifier.add(Dense(output_dim = 256, activation = 'tanh'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 128, activation = 'tanh'))
"""
#classifier.add(Dense(output_dim = 512, activation = 'tanh'))


classifier.add(Dense(output_dim = 9,activation = 'softmax'))
sgd = optimizers.SGD(lr=1e-2)

classifier.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[21]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
	                               shear_range = 0.2,
	                               rotation_range=20,
	                               horizontal_flip = True,
                                   vertical_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
	                               )

image_x, image_y = get_image_size()

training_set = train_datagen.flow_from_directory('C:/Users/Tanmay/Desktop/HandGesture/digit/Train',
	                                             target_size = (image_x,image_y),
	                                             batch_size = 25,
	                                             class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('C:/Users/Tanmay/Desktop/HandGesture/digit/Test',
	                                        target_size = (image_x,image_y),
	                                        batch_size = 25,
	                                        class_mode = 'categorical')


# In[22]:


classifier.fit_generator(training_set,
	          			samples_per_epoch = 500,
	          			nb_epoch = 25,
	          			validation_data = test_set,
	          			nb_val_samples = 50)


# In[30]:



import os
import numpy as np
from matplotlib import pyplot as plt

DATADIR = "C:/Users/Tanmay/Desktop/HandGesture/digit/valid"
Category = []

image_x, image_y = get_image_size()

path=  os.path.join(DATADIR)
for img in os.listdir(path):
    img = cv2.imread(os.path.join(path,img))
    #cv2.imshow("sample:", img)
    plt.imshow(img)
    plt.draw()
    plt.show()  # , plt.draw(), plt.show()
    plt.pause(0.01)
    
    img = cv2.resize(img, (image_x, image_y))
    print(thresh.shape)
    img = np.array(img, dtype=np.float32)
    print(img.shape)
    img = np.reshape(img, (1,image_x, image_y,3))
    print(img.shape)
    pred_probab = classifier.predict(img)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    #cv2.imshow("Image",img)
    print("Pred probab:", pred_probab)
    print("Pred class:", pred_class + 1)
    


# In[31]:


# Open Camera
capture = cv2.VideoCapture(0)

while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = frame[100:300, 100:300]

    # Apply Gaussian blur
    

    # Change color-space from BGR -> HSV
   
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    # Find contour with maximum area
    contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
    hull = cv2.convexHull(contour)

        # Draw contour
    drawing = np.zeros(crop_image.shape, np.uint8)
    cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
        # tips) for all defects
    count_defects = 0
    
    th = cv2.resize(thresh,(image_x, image_y))
    print(th.shape)
    th = np.array(th, dtype=np.float32)
    print(th.shape)
    th = np.reshape(th, (1, image_x, image_y,3))
    pred_probab = classifier.predict(th)[0]
    #pred_probab = s.classifier.predict(thresh)[0]
    #print(pred_probab)
    cv2.putText(frame, pred_probab + 1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),2)
   

    # Show required images
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


# capture.release()
# cv2.destroyAllWindows()

# In[32]:


capture.release()
cv2.destroyAllWindows()


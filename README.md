# DIGIT RECOGNITION
## ABSTRACT
<p>I trained the model to investigate digit recognition using the CNN algorithm with the Mnist dataset. MNIST is one of the common datasets used to train models to recognize handwritten numbers. The data set contains 10 data sets from 0 to 9. I used CNN because it is a very successful algorithm in image classification.</p>
<br>

## DATASET
<p>The MNIST (Modified National Institute of Standards and Technology) dataset is created to recognize individual digits. The MNIST dataset had created by remixing some datasets of the NIST . In the MNIST dataset we have 70000 images of handwritten numbers resized to 28×28 and converted to grayscale </p>

 ![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/bee1ac83-b1fe-4e88-82ca-b6c51ef88d4d)

## How computer sees the data!
![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/8ab23dd8-020c-44e1-a218-f2d75ed5635b)

## How computer sees the normalized data!
![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/d06a687c-6c47-455c-8934-2d75192757f6)

## WHAT IS THE CNN (Convolutional Neural Network)
CNN stands for Convolutional Neural Network. CNN consists of 4 hidden layers which help in extraction of the features from the images and can predict the result. The layers of CNN are (a) Convolutional Layer (b) ReLu Layer (c) Pooling Layer (d) Fully Connected Layer. Reason we are using CNN is because the fundamental favorable position of CNN contrasted with its predecessors is that it consequently recognizes the significant highlights with no human management.

### Convolution Layer 
Convolutional layer is a simple application of a filter which acts as an activation function. What this does is takes a feature from a input image, then filter different features from that image and makes a feature map. Some of the features are location, strength etc. the filter is then moved over the whole image and the value of each pixel is calculated.

![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/ea54f032-7597-4588-a8bf-c4230c060e43)

<p>Sample Filters</p>

![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/4b95002b-151c-4879-ab80-d91a08af718c)
<p>Sample Convolution Layer Feature Maps</p>

![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/b8a8356f-ac31-412f-917f-db1c1d25fef9)
###  Pooling Layer

The main function of this layer is to reduce the image size. This is done to facilitate computational speed and reduce computational cost.What this layer basically does is to take a 2 x 2 matrix and a step of 1 (moving from one pixel to another) and move the window across the entire image. The highest value is taken in each of the windows and this process is repeated for each part of the image. In summary, before the pooling layer we had a 26 x 26 matrix, after the pooling layer the image matrix changed to a 13 x 13 matrix.

![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/0679a1bb-1494-4d83-9f47-f12ddb6c24ab)

<p>Sample Pooling Layer Feature Maps</p>

![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/77991f17-661b-4b8d-aad7-734ae0dc9f5f)

<p>Flattening Layer</p>

![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/e2c43ce5-fb3d-4f99-b5dc-990fbf76d29f)


###	Fully Connected Layer
This is the last layer of CNN. This is the part where the actual classification happens. All the matrix from the pooling layer is stacked up here and put into a single list. 

![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/dafc9d47-20c8-46ff-a20a-1f34460076ff)

## Model Architecture

![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/94f92711-edba-429a-9399-4303e24b8383)


## Image of The Image In Layers

![image](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/3591f580-7ce6-4d17-895e-6d6edd638453)

## Result ==> Accuracy = 0.9921



## CONFUSING MATRIX
![4b959692-3e19-4638-a829-8a3ebdd73d40](https://github.com/AliHanBtmz/DigitRecognation/assets/132774344/9757be9f-4007-443d-8627-6a12e650595b)

 


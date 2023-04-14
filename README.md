
# Project Title: EchoPLAX-Seg

**Advanced Parasternal Long-Axis Echocardiography Segmentation with Deep Learning**

**Description:**

EchoPLAX-Seg is a cutting-edge deep learning project aimed at segmenting medical echocardiography parasternal long-axis (PLAX) view images into six primary heart structures. This innovative deep neural network utilizes a full convolutional network (FCN) architecture inspired by U-Net, and is trained to achieve an efficient and accurate segmentation of echocardiography images, thus enhancing the overall process of clinical assessment and diagnostic accuracy.

**Introduction:**

Echocardiography, a non-invasive imaging technique, plays a critical role in the diagnosis and management of various cardiac conditions. Parasternal long-axis (PLAX) view is an essential perspective in echocardiography, as it offers visualization of six key heart structures: left ventricle, right ventricle, left atrium, aorta, septum, and outer cardiac wall.

In this project, we introduce EchoPLAX-Seg, a powerful deep neural network utilizing a full convolutional network (FCN) architecture inspired by U-Net. Developed with TensorFlow Keras library, the model segments the PLAX view images with high precision, paving the way for automatic calculation of clinical parameters related to cardiac function, such as ejection fraction. EchoPLAX-Seg has demonstrated remarkable intersection-over-union (IOU) accuracy of 89.3% on average (71 test images).

**Color Code:**

-   Yellow: Right Ventricle
-   Light Blue: Septum
-   Dark Blue: Left Ventricle
-   Gray: Aorta
-   Green: Left Atrium
-   Orange: Outer Cardiac Wall

Inspiring by: [@raventan95](https://github.com/raventan95/echo-plax-segmentation)

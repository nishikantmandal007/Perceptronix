## Perceptronix: Dimensional Depth Enigma

Tis was our hackathon project where we have to calculate distance using monocular depth estimation technique , we used depth estimation maps for getting the depth of the various features in a 2D image , using MiDas Machine Learning Model and some custom mathematics to calculate rough distances of the features from camera module.

![Downloader la -64929127d7e37-modified](https://github.com/nishikantmandal007/Perceptronix-Dimensional-Depth-Enigma/assets/113323074/ee1d757b-7fe8-4672-9767-1c50671f28db)


First, we will use a camera to capture video of the scene we're interested in. Then, we'll use a computer program that can estimate the distance of objects in the scene from the camera, even though we're only using one camera. This is called monocular depth estimation.

To figure out if an object is big or small, we'll measure its height, width, and length using the same depth estimation technique. Then we'll compare those measurements to a threshold value to determine whether it's a big or small object.

By doing this, we'll be able to calculate the relative distance of objects from the camera.

USE CASES

Whenever a car would approach into the circular range of a bigger loaded truck an alarm would go off in the car hearing which the driver would carefully drive his car away from the danger zone.
In automatic braking system of a car, the AI of the car can detect the relative distance between itself i.e. the car, and the obstacle in front of or behind it beforehand using this problem solution.

The MIDAS (Mixture of Dense Networks for Single Image Depth Estimation) model for depth estimation is based on a deep convolutional neural network architecture. Specifically, it uses a multi-scale encoder-decoder architecture with skip connections, which is trained to predict the depth map of a given input image. The model is trained using a combination of supervised and unsupervised learning techniques, where the supervised part involves minimizing the mean squared error between the predicted depth map and the ground truth depth map, and the unsupervised part involves enforcing consistency between the predicted depth map and the input image. Overall, the MIDAS model is a state-of-the-art approach for single image depth estimation that has achieved impressive results on a wide range of benchmark datasets.


![unnamed](https://github.com/nishikantmandal007/vision/assets/113323074/5f891d84-e305-49c3-86e2-573b7774bf1f)



![unnamed](https://github.com/nishikantmandal007/vision/assets/113323074/1cb378ac-92dc-4ff5-9e54-7010b5f9b850)


Reason for choosing MiDas was,

- WORKS SMOOTHLY ON LOW END PROCESSOR

- OPEN SOURCE

- REAL TIME PERFORMANCE

- ACCURACY


/*The distance is calculated under the given formula :-

d = (f*b)/(x-x’)

where,d is the distance of object from the optical centre,f is the focal length of optical lens of the webcam,b is the baseline and (x’ , y’) represents the pixel coordinates in the depth map.  */

Architecture: Kitty and Midas use different deep learning architectures. Kitty is based on a ResNet backbone with an encoder-decoder architecture, while Midas uses a lightweight MobileNetV2 backbone with a pyramid fusion module.

Training data: Kitty was trained on the KITTI dataset, which contains real-world driving scenes, while Midas was trained on a combination of synthetic and real-world images.

Output format: The output format of the two models is different. Kitty predicts a single depth map, while Midas produces two maps: a dense depth map and a sparse depth map with fewer details but higher accuracy.

Performance: Both models have achieved state-of-the-art performance on benchmark datasets such as NYU Depth V2 and KITTI, but Midas has been shown to outperform Kitty on some tasks, such as depth estimation for objects at close range.

***Disclaimer The project has not been completed yet as we have to optimize a lot in order to calculate the distacnces accurately .
If anyone wants to contribute feel free to Fork ![git-fork](https://github.com/nishikantmandal007/Perceptronix-Dimensional-Depth-Enigma/assets/113323074/21f910f0-b926-4201-a637-cb4b98848498)
 the project .***

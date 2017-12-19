# Deep Recognition on MNIST Double Digit

## Overview
In this project, we are going to do classification on the MNIST Double Digits(MNISDD).The MNISTDD dataset contains gray scale images of size 64x64. Each image has two MNIST digits (from 0 to 9) randomly placed inside it like in this visualization:![alt text](https://github.com/cjiang2/CMPUT328_MNISTDD/blob/master/example.png) 

## Prerequisites
+ Python3
+ Tensorflow
+ Numpy

## Results
| Model         | All True Accuracy | One True Accuracy|
| ------------- | ----------------- | ---------------- |
| LeNet         |     79.76%        |      96.00%      |
| LeNet_ZCA     |     74.08%        |      95.54%      |
| Shallow       |     96.72%        |      99.74%      |
| Shallow_ZCA   |     87.36%        |      99.8%       | 
| VGG_Like      |     97.7%         |      99.88%      |
| VGG_Like_ZCA  |     98.26%        |      100%        |

## Team Tanh
| Author        |
| ------------- |
| Chen Jiang    |
| Shuyang Li    |
| Jennifer Yuen |
| Yanren Qu     |

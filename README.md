# sign_language_recognition

## Description

Sign language Recognition (SLR) is a breakthrough research field that aims to help bridge the communication gap faced by people with hearing impairments. Although limitations (financial and computational) won't allow a fully functional use for live translation and actual research in the field haven't reached the sustainable commercial applications yet, this proof of concept project's purpose is to demonstrate the progresses made in implementing SLR technology using different Computer Vision technologies. 

## Requirements

* Tensorflow: to build and train the neural network with LSTM layers 
* open-cv-python: computer vision library to work with the webcam and feed the model with keypoints through video stream 
* mediapipe: use of the *Holistic* module to detect keypoints on face and body 
* sklearn: used for evaluation metrics and the train/test split 
* matplotlib: to visualise images 

## Data 

The cost of the various possible data acquisition methods is still unfortunately very high and is one of the main difficulties in producing a scalable SLR application. 

For this project, I used [Spreadthesign](http://www.spreadthesign.com) dictionary with their kind authorization to build the vocabulary required to train the model. I programmed a `data_parser.py` to automatically parse, download and categorize relevent videos. Again, due to memory and computation power concerns, choices had to be made and the vocabulary was restricted to a preselected **INT** definitions. 

## LSTM Model 

## Interface 

## Evaluation

## Demo


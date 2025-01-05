# Blink Detector

## Overview

This project performs blink detection using a pre-trained facial landmark detection model. 
The system calculates the Eye Aspect Ratio (EAR) to determine if a blink has occurred in the video.

## Features

- **Real-time Blink Detection**: Detects blinks in a given video or webcam feed.
- **Eye Aspect Ratio (EAR) Calculation**: Uses a geometric approach to compute EAR for determining blinks.
- **Facial Landmark Detection**: Utilizes dlib's shape_predictor_68_face_landmarks model.

## Installation

**Required Libraries**
```
pip install -r requirements.txt
```

**Requirements File**
```
opencv-python
imutils
dlib
scipy
```

## Explanation of Key Components

**Eye Aspect Ratio (EAR)**

The EAR is calculated using the following formula: Where:

- are the eye landmarks.


**Facial Landmark Detection**

Dlib's pre-trained shape_predictor_68_face_landmarks.dat model is used to extract facial landmarks, 
including eye coordinates.
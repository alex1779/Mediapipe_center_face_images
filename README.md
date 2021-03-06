# Mediapipe Center Face Image
From: https://google.github.io/mediapipe/

## Installation on Windows using Anaconda
```
conda create -n mediapipe_center_face_images -y && conda activate mediapipe_center_face_images && conda install python=3.9.7 -y
git clone https://github.com/alex1779/Mediapipe_center_face_images.git
cd mediapipe_center_face_images
pip install -r requirements.txt
```

## How to run

```
python main.py -i input/example1.jpg
```

## Result
![Image Input](https://github.com/alex1779/Mediapipe_center_face_images/blob/master/imgs/example.jpg)





## How works

```
From the face points generated using mediapipe, use face replacement by triangulating a centered image using cv2 and numpy.
```




## License

Many parts taken from the cpp implementation from github.com/google/mediapipe

Copyright 2020 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.







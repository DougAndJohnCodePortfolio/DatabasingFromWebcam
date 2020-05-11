# Databasing From A Webcam

Doug and John's solution to capturing better data from a live environment
when using the machine learning tool face_recognition and the image capture tool OpenCV.

This project follows in the steps of the  [face_recognition] and [OpenCV] tutorials.

This code:
* Creates a folder called Data next to where the Python script is run
* Starts the default webcam
* Uses previous webcam frame to determine if current frame is reliable data
* Captures cleaner data from webcam
* Saves data in machine-friendly bits and human-friendly .jpg files


A fancier presentation can be found on the other side of our [GitHub].

### Linux: Ubuntu 20.04 Environment Setup
```
sudo add-apt-repository universe
sudo apt install cmake
sudo apt install python3-pip
pip3 install opencv-python
pip3 install face_recognition
```

### Run Code
```
python3 DatabasingFromWebcam.py
```

### Mac / Windows support

Coming soon!

   [face_recognition]: <https://github.com/ageitgey/face_recognition>
   [OpenCV]: <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html>
   [GitHub]
# GazeFollow

Reimplementation in python of matlab code from
http://gazefollow.csail.mit.edu/

# Installation
Download model weights (.mat, .prototxt and .caffemodel files) from
http://gazefollow.csail.mit.edu/download.html
and put them into ./utils/

Dependencies (tested with Python 3.6 but later versions should work):
- Numpy
- Opencv 4
- Scipy (for reading matlab files within python

# Usage

Visualize GazeFollow dataset
```
python test_gazefollow_reader.py <path/to/gazefollow/train_annotations.txt>
```

Test the network on a custom picture, manually giving eye position (x and y in (0,1))

```
python test_gazefollow_network.py <path/to/picture.png> <eye_x> <eye_y>
```

Test the network on a random pictures from the gazefollow dataset
```
python test_gazefollow_network_2.py <path/to/gazefollow/train_annotations.txt>
```

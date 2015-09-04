# HandNet
Code repository for visualizing and manipulating HandNet

Current project page is [here](http://www.cs.technion.ac.il/~twerd/HandNet/) 
Data needs to be downloaded from here (into respective folders):
[Train (12.5 GB) ](http://www.cs.technion.ac.il/~twerd/TrainData.rar)
[Test (627 MB) ](http://www.cs.technion.ac.il/~twerd/TestData.zip)
[Validation (174 MB)](http://www.cs.technion.ac.il/~twerd/ValidationData.zip)

## Visualize data
For Matlab or Octave users just run DisplaySession.m and change
the desired visualization accordingly. 

Current examples dont include Python. However the data is in MAT format
and reading it with Python is straightforward:

For example (assuming you downloaded the validation data to Validation folder):
```python
import scipy.io
data = scipy.io.loadmat('Validation/Data_0000000.mat')
```
## Derotation
TODO ...

## Cropping to bounding box and generating HDF5 files 
TODO ...

## Performing training 
TODO ...

## How to evaluate another method on test data
TODO ...


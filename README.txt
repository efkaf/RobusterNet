Requirements


Python 2.7
Keras 2.2.2
TensorFlow 1.8.0


Volterra Layer

Import the layer
from quadraticLayer import QuadraticLayer

Plug the layer into a Keras model like any other layer in Keras
x1 = QuadraticLayer(64, (3, 3), activation='relu', padding="SAME", strides=(1, 1))(img_input)


Datasets

Download CASIA and CoMoFoD full datasets from https://github.com/isi-vista/BusterNet/tree/master/Data under the directory data/

Reproducing the results

Run fig6.py to reproduce Fig.6
Run busterDemoCasia.py and busterDemoCoMoFoD.py to reproduce Tables II and III

Evaluation code and BusterNet weights from https://github.com/isi-vista/BusterNet


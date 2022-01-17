Purpose
===
I forked this repository to convert WHENet.h5 to tnn model files run in TNN framework in Android and iOS.

How to
===
1. Merged PR#12 into master
2. Change demo.py to use whenet_tf.py
3. Add k2o.py to convert WHENet.h5 to WHENet.onnx
4. Add tests.py to run predicts on images under Sample/ by h5 and onnx models to prove onnx model is right


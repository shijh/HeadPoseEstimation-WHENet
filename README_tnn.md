Purpose
===
I forked this repository to convert WHENet.h5 to tnn model files run in TNN framework in Android and iOS.

How to
===
1. Merged PR#12 into master
2. Change demo.py to use whenet_tf.py
3. Add k2o.py to convert WHENet.h5 to WHENet.onnx
4. Add tf2o.py to convert WHENet.h5 to WHENet_tf.onnx (this WHENet_tf.onnx can be converted to TNN successfully)
5. Add tests.py to run predicts on images under Sample/ by h5 and onnx models to prove onnx model is right
6. Install protobuf 3.5.0 from source (on Ubuntu 18.04):
```
git clone https://github.com/protocolbuffers/protobuf
cd protobuf
git checkout tags/v3.5.0

git submodule update --init --recursive
./autogen.sh
./configure
make
make check
sudo make install
sudo ldconfig
protoc --version
```
7. Install TNN project from source (run on Python 3.7 on Ubuntu 18.04): 
```
git clone https://github.com/tencent/TNN
cd TNN/tools/onnx2tnn/onnx-converter/
./build.sh
python3 onnx2tnn.py WHENet_tf.onnx -version=v1.0 -optimize=0 -half=1 -o ./ -input_shape input:1,224,224,3
```

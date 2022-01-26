import tf2onnx
import onnx
from whenet_tf import WHENetTF
import cv2
import numpy as np
from utils import draw_axis


def crop_and_predict(img_path, head_box, tf_model):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_min, y_min, x_max, y_max = head_box
    img_rgb = img_rgb[y_min:y_max, x_min:x_max]
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = np.expand_dims(img_rgb, axis=0)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 1)
    yaw, pitch, roll = tf_model.get_angle(img_rgb)
    draw_axis(img, yaw, pitch, roll, tdx=(x_min + x_max) / 2, tdy=(y_min + y_max) / 2, size=abs(x_max - x_min))
    cv2.imshow('output', img)
    cv2.waitKey(5000)


def whenettf_to_onnx(img_path, head_box, tf_model):
    # run keras predict
    crop_and_predict(img_path, head_box, tf_model)

    # and then convert
    onnx_model_proto, _ = tf2onnx.convert.from_keras(tf_model.model, opset=10)
    onnx.save_model(onnx_model_proto, 'WHENet_tf.onnx')


if __name__ == "__main__":
    tf_model = WHENetTF('WHENet.h5')
    root = 'Sample/'
    print(tf_model.model.summary())
    with open('Sample/bbox.txt', 'r') as f:
        lines = f.readlines()

    # only run the 1st line
    for line in lines:
        filename, head_box = line.split(',')
        head_box = head_box.split(' ')
        head_box = [int(b) for b in head_box]
        whenettf_to_onnx(root + filename, head_box, tf_model)
        break

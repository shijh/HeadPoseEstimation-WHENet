import numpy as np
import cv2
from whenet_tf import WHENetTF
from utils import draw_axis
import onnxruntime as rt
from utils import softmax


def h5_crop_and_pred(img_path, bbox, model):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_min, y_min, x_max, y_max = bbox
    img_rgb = img_rgb[y_min:y_max, x_min:x_max]
    img_rgb = cv2.resize(img_rgb,(224,224))
    print("h5 img_rgb shape: ", img_rgb.shape)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    print("h5 img_rgb shape: ", img_rgb.shape)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 1)
    yaw, pitch, roll = model.get_angle(img_rgb)
    print("h5 predict results for ", img_path, ":")
    print("  yaw:   ", yaw)
    print("  pitch: ", pitch)
    print("  roll:  ", roll)
    draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size=abs(x_max-x_min))
    cv2.imshow('output', img)
    cv2.waitKey(5000)


def onnx_crop_and_pred(img_path, bbox, sess, idx_tensor_yaw, idx_tensor):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_min, y_min, x_max, y_max = bbox
    img_rgb = img_rgb[y_min:y_max, x_min:x_max]
    img_rgb = cv2.resize(img_rgb, (224, 224))
    print("onnx img_rgb shape: ", img_rgb.shape)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_rgb = img_rgb / 255
    img_rgb = (img_rgb - mean) / std
    img_rgb = img_rgb.astype(np.float32)
    print("onnx img_rgb shape: ", img_rgb.shape)
    input_name = sess.get_inputs()[0].name
    res = sess.run(None, {input_name: img_rgb})
    # print('onnxruntime:\n', res)
    yaw_predicted = softmax(res[0])
    pitch_predicted = softmax(res[1])
    roll_predicted = softmax(res[2])
    # print('onnxruntime yaw softmax:\n', yaw_predicted)
    # print('onnxruntime pitch softmax:\n', pitch_predicted)
    # print('onnxruntime roll softmax:\n', roll_predicted)
    # print('idx_tensor_yaw: ', idx_tensor_yaw)
    # print('idx_tensor: ', idx_tensor)
    yaw_predicted = np.sum(yaw_predicted * idx_tensor_yaw, axis=1) * 3 - 180
    pitch_predicted = np.sum(pitch_predicted * idx_tensor, axis=1) * 3 - 99
    roll_predicted = np.sum(roll_predicted * idx_tensor, axis=1) * 3 - 99
    print("onnx predict results for ", img_path, ":")
    print("  yaw:   ", yaw_predicted)
    print("  pitch: ", pitch_predicted)
    print("  roll:  ", roll_predicted)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 1)
    draw_axis(img, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size=abs(x_max-x_min))
    cv2.imshow('output', img)
    cv2.waitKey(5000)


if __name__ == "__main__":
    model = WHENetTF('WHENet.h5')
    root = 'Sample/'
    print(model.model.summary())

    sess = rt.InferenceSession('WHENet.onnx')

    with open('Sample/bbox.txt', 'r') as f:
        lines = f.readlines()

    for line in lines:
        filename, bbox = line.split(',')
        bbox = bbox.split(' ')
        bbox = [int(b) for b in bbox]
        h5_crop_and_pred(root+filename, bbox, model)
        print()
        onnx_crop_and_pred(root+filename, bbox, sess, model.idx_tensor_yaw, model.idx_tensor)
        print("\n-------------------")

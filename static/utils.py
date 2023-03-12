import os
import io
import cv2
import json
import onnx
import math
import base64
import numpy as np
import onnxruntime as ort

from PIL import Image

STATIC_PATH: str = "static"

ort.set_default_logger_severity(3)

#####################################################################################################

class YoloV3(object):
    def __init__(self, model_type: str="tiny") -> None:
        self.size: int = 416
        self.model_type = model_type
        self.classes = json.load(open("static/labels.json", "r"))

        if self.model_type == "tiny": self.path: str = os.path.join(STATIC_PATH, f"models/yolo-v3-t.onnx")
    
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        scale = min(self.size / w, self.size / h)

        nh, nw = math.ceil(h * scale), math.ceil(w * scale)

        hh: int = (self.size - nh) // 2
        ww: int = (self.size - nw) // 2

        image = cv2.resize(src=image, dsize=(nw, nh), interpolation=cv2.INTER_AREA)
        new_image = np.ones((self.size, self.size, 3), dtype=np.uint8) * 128
        
        if abs(nh-(self.size - 2*hh)) == 1: new_image[hh:self.size-hh-1, ww:self.size-ww, :] = image
        elif abs(nw-(self.size - 2*ww)) == 1: new_image[hh:self.size-hh, ww:self.size-ww-1, :] = image
        else: new_image[hh:self.size-hh, ww:self.size-ww, :] = image

        new_image = new_image.transpose(2, 0, 1).astype("float32")
        new_image /= 255
        new_image = np.expand_dims(new_image, axis=0)
        return new_image
    
    def infer(self, image: np.ndarray) -> tuple:
        
        image_h, image_w, _ = image.shape
        image = self.preprocess(image=image)

        input = {
            self.ort_session.get_inputs()[0].name : image,
            self.ort_session.get_inputs()[1].name : np.array([image_h, image_w], dtype=np.float32).reshape(1, 2),
        }
        
        boxes, scores, indices = self.ort_session.run(None, input)

        out_boxes, out_scores, out_classes = [], [], []

        if len(indices[0]) != 0:
            for idx_ in indices[0]:
                out_classes.append(idx_[1])
                out_scores.append(scores[tuple(idx_)])
                idx_1 = (idx_[0], idx_[2])
                out_boxes.append(boxes[idx_1])
            
            x1, y1, x2, y2 = int(out_boxes[0][1]), int(out_boxes[0][0]), int(out_boxes[0][3]), int(out_boxes[0][2])
            
            return self.classes[str(out_classes[0])], out_scores[0], (x1, y1, x2, y2)
        return None, None, None
    
#####################################################################################################

class YoloV6(object):
    def __init__(self, model_type: str="nano") -> None:
        self.size: int = 640
        self.model_type = model_type

        if model_type == "tiny": self.path: str  = os.path.join(STATIC_PATH, f"models/yolo-v6-t.onnx")
        if model_type == "small": self.path: str = os.path.join(STATIC_PATH, f"models/yolo-v6-s.onnx")
        if model_type == "nano": self.path: str  = os.path.join(STATIC_PATH, f"models/yolo-v6-n.onnx")

        self.classes = json.load(open("static/labels.json", "r"))
    
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = image / 255
        image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image.astype("float32")
    
    def process_result(self, result: np.ndarray, im_w: int, im_h: int) -> tuple:
        result = result[0]
        probabilities = result[:, 5:]
        label_index = np.argmax(np.max(probabilities, 0))
        score = np.max(np.max(probabilities, 0))
        box_index = np.argmax(probabilities[:, label_index] == np.max(probabilities[:, label_index]))
        box = result[box_index, :4]
        cx = int(box[0] * (im_w / 640))
        cy = int(box[1] * (im_h / 640))

        w = int(box[2] * (im_w / 640))
        h = int(box[3] * (im_h / 640))

        x1 = cx - (w // 2)
        y1 = cy - (h // 2)

        x2 = cx + (w // 2)
        y2 = cy + (h // 2)

        return self.classes[str(label_index)], score, (x1, y1, x2, y2)

    def infer(self, image: np.ndarray) -> tuple:

        image_h, image_w, _ = image.shape

        input = {self.ort_session.get_inputs()[0].name : self.preprocess(image)}
        result = self.ort_session.run(None, input)
        label, score, (x1, y1, x2, y2) = self.process_result(result[0], image_w, image_h)

        return label, score, (x1, y1, x2, y2)

#####################################################################################################

class YoloV7(object):
    def __init__(self, model_type: str="nano") -> None:
        self.size: int = 640
        self.model_type = model_type

        if model_type == "tiny": self.path: str  = os.path.join(STATIC_PATH, f"models/yolo-v7-t.onnx")

        self.classes = json.load(open("static/labels.json", "r"))

        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = image / 255
        image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image.astype("float32")
    
    def infer(self, image: np.ndarray) -> tuple:
        im_h, im_w, _ = image.shape

        input = {self.ort_session.get_inputs()[0].name : self.preprocess(image)}
        result = self.ort_session.run(None, input)
        result = result[0][0]

        box = result[1:5]
        label_index = int(result[-2])
        score = result[-1]

        x1 = int(box[0] * im_w / 640)
        y1 = int(box[1] * im_h / 640)
        x2 = int(box[2] * im_w / 640)
        y2 = int(box[3] * im_h / 640)

        return self.classes[str(label_index)], score, (x1, y1, x2, y2)

#####################################################################################################

class YoloV8(object):
    def __init__(self, model_type: str="nano") -> None:
        self.size: int = 640
        self.model_type = model_type

        if model_type == "small": self.path: str  = os.path.join(STATIC_PATH, f"models/yolo-v8-s.onnx")
        if model_type == "nano": self.path: str  = os.path.join(STATIC_PATH, f"models/yolo-v8-n.onnx")

        self.classes = json.load(open("static/labels.json", "r"))

        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = image / 255
        image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image.astype("float32")
    
    def infer(self, image: np.ndarray) -> tuple:
        im_h, im_w, _ = image.shape

        input = {self.ort_session.get_inputs()[0].name : self.preprocess(image)}
        result = self.ort_session.run(None, input)
        result = result[0][0]

        boxes = result[0:4]
        label_index = np.argmax(np.max(result[4:], axis=1))
        score = np.max(np.max(result[4:], axis=1))

        best_box_index = np.argmax(result[4:], axis=1)[label_index]
        box = boxes[:, best_box_index].astype("int32")

        cx = int(box[0] * im_w / 640)
        cy = int(box[1] * im_h / 640)
        w  = int(box[2] * im_w / 640)
        h  = int(box[3] * im_h / 640)

        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = cx + w // 2
        y2 = cy + h // 2

        return self.classes[str(label_index)], score, (x1, y1, x2, y2)

#####################################################################################################

def decode_image(imageData: str) -> np.ndarray:
    header, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    return header, image


def encode_image_to_base64(header: str="data:image/png;base64", image: np.ndarray=None) -> str:
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".jpeg", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'", "").replace("'", "")
    imageData = header + "," + imageData
    return imageData

#####################################################################################################
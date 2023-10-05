import cv2 as cv
import json
import joblib
import base64
from wavelet import w2d
import numpy as np




__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_img (img_b64_data, file_path=None):
    images = get_cropped_image(file_path, img_b64_data)

    result = []

    for img in images:
        scale_raw_img = cv.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scale_har_img = cv.resize(img_har, (32, 32))
        combine_img = np.vstack((scale_raw_img.reshape(32 * 32 * 3, 1), scale_har_img.reshape(32 * 32, 1)))
        len_image_array = 32 * 32 * 3 + 32 *32
        final = combine_img.reshape(1, len_image_array).astype(float)

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final)*100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result


def load_artifacts():
    print("Loading saved artifacts.. start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/post_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts..done")

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_cv_img_from_base64_string(b64str):

    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return image

def get_cropped_image(image_path, img_b64_data):
    face_cascade = cv.CascadeClassifier('../model/Haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('../model/Haarcascades/haarcascade_eye.xml')

    if image_path:
        image = cv.imread(image_path)
    else:
        image = get_cv_img_from_base64_string(img_b64_data)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_imgb64():
    with open("imgb64.txt") as f:
        return f.read()



if __name__ == "__main__":
    load_artifacts()
    # print(classify_img(get_imgb64(), None))
    print(classify_img(None, "./test_image/messi2.png"))
    print(classify_img(None, "./test_image/messi1.jpg"))
    print(classify_img(None, "./test_image/williams2.jpg"))
    print(classify_img(None, "./test_image/russel3.jpg"))


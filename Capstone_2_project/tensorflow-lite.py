#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


preprocessor = create_preprocessor("xception", target_size=(299, 299))

interpreter = tflite.Interpreter(model_path="dog_cat_model.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

classes = ['cats', 'dogs']

# url = https://drive.google.com/file/d/1kzQfxYU1SU3EbJj4IKB9lb0jufVihPST/view?usp=drive_link


def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)

    return dict(zip(classes, pred[0]))


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result

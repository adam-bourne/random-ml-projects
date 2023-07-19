import os
from transformers import pipeline
from PIL import Image
import logging

image_fixtures_path = os.path.join(os.path.dirname(__file__), "image_files")
batch_predict = False

###########################################################
# CONFIG
###########################################################


def create_image_batch(image_dir):
    """takes a path to a directory of images and creates a dataframe of image objects"""

    image_list = []
    image_file_names = []

    for image_file_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, image_file_name)
        try:
            img = Image.open(img_path)
            image_file_names.append(image_file_name)
            image_list.append(img)
        except Exception as e:
            logging.info(f"Unsupported file type, please use an image file: {e}")

    return image_file_names, image_list


def make_predictions(image_file_names, image_list, batch=False):
    """takes image objects and image names and makes predictions using the pipeline, return results as a dictionary"""

    classifier = pipeline("image-classification", model="adam-bourne/food-classifier")
    predictions = {}

    # for small numbers of predictions batching is often slower using pipelines, so default will be individual
    if not batch:
        for i, image in enumerate(image_list):
            results = classifier(image)
            results_dict = {"label": results[0]["label"], "score": results[0]["score"]}
            predictions[image_file_names[i]] = results_dict
    else:
        results = classifier(image_list, batch_size=len(image_list))
        for i, result in enumerate(results):
            results_dict = {
                "label": results[i][0]["label"],
                "score": results[i][0]["score"],
            }
            predictions[image_file_names[i]] = results_dict

    return predictions


def classify_food_images(image_dir):
    """combine the stages of creating our batch and generating predictions"""

    image_file_names, image_list = create_image_batch(image_dir)
    predictions = make_predictions(image_file_names, image_list, batch=batch_predict)

    return predictions


if __name__ == "__main__":
    output_predictions = classify_food_images(image_fixtures_path)
    print(output_predictions)

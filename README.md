# random-ml-projects
miscellaneous ML ideas that don't really fit anywhere else.

## 1. food-classifier
This is an image classification pipeline which uses Hugging Face Transformers to fine-tune a 'google/vit-base-patch16-224-in21k' model on images of food. There are 101 different food classes it is trained on, and two example images are provided in `food_classifier/image_files/`. For more info see the [model card](https://huggingface.co/adam-bourne/food-classifier) on Hugging Face.

#### Usage
cd to the `food_classifier` directory and run `python3 food_classifier.py` after adding the images you want to test to `food_classifier/image_files/`. Results are a dictionary of image names and corresponding labels and scores

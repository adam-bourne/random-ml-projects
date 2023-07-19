import os
import pytest
from PIL import Image
from food_classifier import create_image_batch, make_predictions, classify_food_images

image_fixtures_path = os.path.join(os.path.dirname(__file__), "image_files")

###########################################################
# CONFIG
###########################################################


def test_it_creates_image_batch():
    image_list_for_test = [
        Image.open(image_path)
        for image_path in [
            os.path.join(image_fixtures_path, f)
            for f in os.listdir(image_fixtures_path)
        ]
    ]
    image_file_names, image_list = create_image_batch(image_fixtures_path)

    assert image_list, image_file_names == (
        image_list_for_test,
        ["beignets.jpeg", "bread.jpeg"],
    )


@pytest.mark.parametrize("batch", [True, False])
def test_it_makes_predictions(batch):
    image_file_names, image_list = create_image_batch(image_fixtures_path)
    predictions = make_predictions(image_file_names, image_list, batch)

    assert predictions == {
        "beignets.jpeg": {"label": "beignets", "score": 0.979089081287384},
        "bread.jpeg": {"label": "bruschetta", "score": 0.9325453042984009},
    }


def test_it_classifies_food_images():
    predictions = classify_food_images(image_fixtures_path)

    assert predictions == {
        "beignets.jpeg": {"label": "beignets", "score": 0.979089081287384},
        "bread.jpeg": {"label": "bruschetta", "score": 0.9325453042984009},
    }

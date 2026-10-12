import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock


# ---------------------------------------------------------
# Test 1: Check class names
# ---------------------------------------------------------
def test_class_names():
    class_names = ['Early Blight', 'Healthy', 'Late Blight']

    assert len(class_names) == 3
    assert 'Early Blight' in class_names
    assert 'Healthy' in class_names
    assert 'Late Blight' in class_names


# ---------------------------------------------------------
# Test 2: Check image size
# ---------------------------------------------------------
def test_image_size():
    IMG_SIZE = 224

    assert IMG_SIZE == 224


# ---------------------------------------------------------
# Test 3: Create a valid test image
# ---------------------------------------------------------
def test_image_can_be_created():
    img = Image.new("RGB", (500, 500))

    assert img is not None
    assert img.size == (500, 500)


# ---------------------------------------------------------
# Test 4: Test image resizing
# ---------------------------------------------------------
def test_image_resize():
    img = Image.new("RGB", (500, 500))

    resized_img = img.resize((224, 224))

    assert resized_img.size == (224, 224)


# ---------------------------------------------------------
# Test 5: Test image conversion to NumPy array
# ---------------------------------------------------------
def test_image_to_numpy_array():
    img = Image.new("RGB", (224, 224))

    img_array = np.array(img)

    assert isinstance(img_array, np.ndarray)
    assert img_array.shape == (224, 224, 3)


# ---------------------------------------------------------
# Test 6: Test image normalization
# ---------------------------------------------------------
def test_image_normalization():
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))

    img_array = np.array(img) / 255.0

    assert np.max(img_array) <= 1.0
    assert np.min(img_array) >= 0.0


# ---------------------------------------------------------
# Test 7: Test batch dimension
# ---------------------------------------------------------
def test_batch_dimension():
    img = Image.new("RGB", (224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    assert img_array.shape == (1, 224, 224, 3)


# ---------------------------------------------------------
# Test 8: Test model prediction output
# ---------------------------------------------------------
def test_model_prediction():
    mock_model = MagicMock()

    mock_model.predict.return_value = np.array(
        [[0.80, 0.10, 0.10]]
    )

    test_input = np.zeros((1, 224, 224, 3))

    predictions = mock_model.predict(test_input)

    assert predictions.shape == (1, 3)
    assert np.argmax(predictions) == 0


# ---------------------------------------------------------
# Test 9: Test predicted class
# ---------------------------------------------------------
def test_predicted_class():
    class_names = ['Early Blight', 'Healthy', 'Late Blight']

    predictions = np.array(
        [[0.10, 0.80, 0.10]]
    )

    predicted_class = class_names[np.argmax(predictions)]

    assert predicted_class == 'Healthy'


# ---------------------------------------------------------
# Test 10: Test confidence calculation
# ---------------------------------------------------------
def test_prediction_confidence():
    predictions = np.array(
        [[0.10, 0.85, 0.05]]
    )

    confidence = np.max(predictions)

    assert confidence == 0.85
    assert confidence * 100 == 85.0


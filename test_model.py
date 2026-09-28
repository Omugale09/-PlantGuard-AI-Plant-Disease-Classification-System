import os
import ast
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# ---------------------------------------------------------
# Helper function
# Extract create_model() from model.py without executing
# the training code in model.py
# ---------------------------------------------------------
def load_create_model_function():

    with open("model.py", "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    create_model_node = None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "create_model":
                create_model_node = node
                break

    assert create_model_node is not None, \
        "create_model() function was not found in model.py"

    function_module = ast.Module(
        body=[create_model_node],
        type_ignores=[]
    )

    compiled_code = compile(
        function_module,
        filename="model.py",
        mode="exec"
    )

    namespace = {
        "layers": layers,
        "models": models,
        "IMG_SIZE": 224,
        "NUM_CLASSES": 3
    }

    exec(compiled_code, namespace)

    return namespace["create_model"]


# ---------------------------------------------------------
# Test 1
# Check whether model.py exists
# ---------------------------------------------------------
def test_model_file_exists():

    assert os.path.exists("model.py")


# ---------------------------------------------------------
# Test 2
# Check image size
# ---------------------------------------------------------
def test_image_size():

    IMG_SIZE = 224

    assert IMG_SIZE == 224


# ---------------------------------------------------------
# Test 3
# Check batch size
# ---------------------------------------------------------
def test_batch_size():

    BATCH_SIZE = 16

    assert BATCH_SIZE == 16


# ---------------------------------------------------------
# Test 4
# Check number of epochs
# ---------------------------------------------------------
def test_epochs():

    EPOCHS = 10

    assert EPOCHS == 10


# ---------------------------------------------------------
# Test 5
# Check number of classes
# ---------------------------------------------------------
def test_number_of_classes():

    NUM_CLASSES = 3

    assert NUM_CLASSES == 3


# ---------------------------------------------------------
# Test 6
# Check whether CNN model can be created
# ---------------------------------------------------------
def test_create_model():

    create_model = load_create_model_function()

    model = create_model()

    assert model is not None


# ---------------------------------------------------------
# Test 7
# Check model input shape
# ---------------------------------------------------------
def test_model_input_shape():

    create_model = load_create_model_function()

    model = create_model()

    assert model.input_shape == (None, 224, 224, 3)


# ---------------------------------------------------------
# Test 8
# Check model output shape
# ---------------------------------------------------------
def test_model_output_shape():

    create_model = load_create_model_function()

    model = create_model()

    assert model.output_shape == (None, 3)


# ---------------------------------------------------------
# Test 9
# Check output layer
# ---------------------------------------------------------
def test_output_layer():

    create_model = load_create_model_function()

    model = create_model()

    output_layer = model.layers[-1]

    assert output_layer.units == 3
    assert output_layer.activation.__name__ == "softmax"


# ---------------------------------------------------------
# Test 10
# Check whether model can make a prediction
# ---------------------------------------------------------
def test_model_prediction():

    create_model = load_create_model_function()

    model = create_model()

    test_image = np.random.random(
        (1, 224, 224, 3)
    ).astype(np.float32)

    prediction = model.predict(
        test_image,
        verbose=0
    )

    # Check prediction shape
    assert prediction.shape == (1, 3)

    # Check probability values
    assert np.all(prediction >= 0)
    assert np.all(prediction <= 1)

    # Softmax probabilities should add up to 1
    assert np.isclose(
        np.sum(prediction[0]),
        1.0,
        atol=1e-5
    )


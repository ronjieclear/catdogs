import os
import uuid

os.environ["MPLCONFIGDIR"] = ".matplotlib"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename


IMAGE_SIZE = 128
WEIGHTS_FILE = "dog_cat_tensorflow_model.weights.h5"
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_DIAGRAM_FILE = "tensorflow_actual_model_diagram.png"


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def build_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.12)
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_layer"),
        data_augmentation,

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
            name="hidden_conv_layer_1"
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling_1"),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
            name="hidden_conv_layer_2"
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling_2"),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
            name="hidden_conv_layer_3"
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling_3"),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
            name="hidden_conv_layer_4"
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling_4"),

        tf.keras.layers.Flatten(name="flatten_layer"),
        tf.keras.layers.Dense(128, activation="relu", name="hidden_dense_layer"),
        tf.keras.layers.Dropout(0.5, name="dropout_layer"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")
    ])

    return model


def prepare_image(file_path):
    image = Image.open(file_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)


def get_layer_rows(model):
    rows = []

    for layer in model.layers:
        try:
            output_shape = str(layer.output.shape)
        except Exception:
            output_shape = "Not built"

        rows.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": output_shape,
                "params": layer.count_params()
            }
        )

    return rows


def get_weight_rows(model):
    rows = []

    for layer in model.layers:
        weights = layer.get_weights()

        if not weights:
            rows.append(
                {
                    "layer": layer.name,
                    "shape": "-",
                    "mean": "-",
                    "std": "-",
                    "minimum": "-",
                    "maximum": "-",
                    "sample": "No learned weights"
                }
            )
            continue

        flattened_parts = []
        shapes = []

        for weight_array in weights:
            flattened_parts.append(weight_array.flatten())
            shapes.append(str(weight_array.shape))

        flattened = np.concatenate(flattened_parts)
        sample_values = flattened[:8]
        sample = ", ".join(f"{value:.4f}" for value in sample_values)

        rows.append(
            {
                "layer": layer.name,
                "shape": " | ".join(shapes),
                "mean": f"{np.mean(flattened):.5f}",
                "std": f"{np.std(flattened):.5f}",
                "minimum": f"{np.min(flattened):.5f}",
                "maximum": f"{np.max(flattened):.5f}",
                "sample": sample
            }
        )

    return rows


def allowed_file(file_name):
    extension = os.path.splitext(file_name)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def summarize_layer_weights(layer):
    weights = layer.get_weights()

    if not weights:
        return {
            "has_weights": False,
            "mean": 0.0,
            "strength": 0.0
        }

    flattened_parts = []

    for weight_array in weights:
        flattened_parts.append(weight_array.flatten())

    flattened = np.concatenate(flattened_parts)

    return {
        "has_weights": True,
        "mean": float(np.mean(flattened)),
        "strength": float(np.mean(np.abs(flattened)))
    }


def generate_model_diagram(model, file_name):
    layer_names = [
        "Input\n128x128x3",
        "Augment",
        "Hidden Conv2D 1\n32 filters",
        "MaxPool 1",
        "Hidden Conv2D 2\n64 filters",
        "MaxPool 2",
        "Hidden Conv2D 3\n128 filters",
        "MaxPool 3",
        "Hidden Conv2D 4\n128 filters",
        "MaxPool 4",
        "Flatten",
        "Hidden Dense\n128 neurons",
        "Dropout",
        "Output\nCat/Dog"
    ]

    layer_objects = [None] + list(model.layers)
    x_positions = np.linspace(0.5, 13.5, len(layer_names))
    y_position = 1.0

    weight_strengths = []

    for layer in model.layers:
        summary = summarize_layer_weights(layer)

        if summary["has_weights"]:
            weight_strengths.append(summary["strength"])

    max_strength = max(weight_strengths) if weight_strengths else 1.0

    plt.figure(figsize=(20, 6))
    plt.title(
        "Actual TensorFlow CNN Model\n"
        "Blue connection = positive average learned weight, red = negative, gray = no learned weights",
        fontsize=16
    )

    for index in range(len(layer_names) - 1):
        next_layer = layer_objects[index + 1]

        if next_layer is None:
            line_color = "#7a8494"
            line_width = 2
            label = ""
        else:
            summary = summarize_layer_weights(next_layer)

            if not summary["has_weights"]:
                line_color = "#7a8494"
                line_width = 2
                label = "no weights"
            else:
                if summary["mean"] >= 0:
                    line_color = "royalblue"
                else:
                    line_color = "tomato"

                line_width = 1.5 + (summary["strength"] / max_strength) * 5
                label = f"mean {summary['mean']:.4f}"

        plt.plot(
            [x_positions[index], x_positions[index + 1]],
            [y_position, y_position],
            color=line_color,
            linewidth=line_width,
            alpha=0.72,
            zorder=1
        )

        if label:
            plt.text(
                (x_positions[index] + x_positions[index + 1]) / 2,
                y_position + 0.16,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4b5565"
            )

    for index, layer_name in enumerate(layer_names):
        layer = layer_objects[index]

        if index == 0:
            fill_color = "#f8fbff"
            param_text = "image"
            shape_text = ""
        elif layer is None:
            fill_color = "#f8fbff"
            param_text = ""
            shape_text = ""
        else:
            if "hidden" in layer.name:
                fill_color = "#eef4ff"
            elif layer.name == "output_layer":
                fill_color = "#fff3ed"
            else:
                fill_color = "#f7f8fb"

            param_text = f"{layer.count_params():,} params"

            try:
                shape_text = str(layer.output.shape)
            except Exception:
                shape_text = ""

        rectangle = plt.Rectangle(
            (x_positions[index] - 0.43, y_position - 0.31),
            0.86,
            0.62,
            facecolor=fill_color,
            edgecolor="#1d2430",
            linewidth=1.4,
            zorder=2
        )
        plt.gca().add_patch(rectangle)

        plt.text(
            x_positions[index],
            y_position + 0.06,
            layer_name,
            ha="center",
            va="center",
            fontsize=9,
            zorder=3
        )
        plt.text(
            x_positions[index],
            y_position - 0.17,
            param_text,
            ha="center",
            va="center",
            fontsize=7,
            color="#5d6675",
            zorder=3
        )

        if shape_text:
            plt.text(
                x_positions[index],
                y_position - 0.53,
                shape_text.replace("None, ", ""),
                ha="center",
                va="center",
                fontsize=7,
                color="#5d6675"
            )

    plt.plot([], [], color="royalblue", linewidth=4, label="Positive average weight")
    plt.plot([], [], color="tomato", linewidth=4, label="Negative average weight")
    plt.plot([], [], color="#7a8494", linewidth=4, label="No learned weights")
    plt.legend(loc="upper right")

    plt.xlim(0, 12)
    plt.ylim(0.15, 1.85)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(file_name, dpi=200)
    plt.close()


model = build_model()
model.load_weights(WEIGHTS_FILE)

test_accuracy = 0.675
report = {
    "Cat": {"f1-score": 0.6632},
    "Dog": {"f1-score": 0.6860},
    "macro avg": {"f1-score": 0.6746},
    "weighted avg": {"f1-score": 0.6746}
}
matrix = [
    [64, 36],
    [29, 71]
]

layer_rows = get_layer_rows(model)
weight_rows = get_weight_rows(model)
generate_model_diagram(model, MODEL_DIAGRAM_FILE)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please choose an image first."
        elif not allowed_file(uploaded_file.filename):
            error = "Please upload a JPG or PNG image."
        else:
            original_name = secure_filename(uploaded_file.filename)
            extension = os.path.splitext(original_name)[1].lower()
            saved_name = f"{uuid.uuid4().hex}{extension}"
            saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
            uploaded_file.save(saved_path)

            image_array = prepare_image(saved_path)
            dog_probability = model.predict(image_array, verbose=0)[0][0]

            if dog_probability >= 0.5:
                predicted_label = "Dog"
                confidence = dog_probability
            else:
                predicted_label = "Cat"
                confidence = 1 - dog_probability

            result = {
                "image_url": f"/static/uploads/{saved_name}",
                "predicted_label": predicted_label,
                "confidence": f"{confidence * 100:.2f}%",
                "dog_probability": f"{dog_probability * 100:.2f}%",
                "cat_probability": f"{(1 - dog_probability) * 100:.2f}%"
            }

    return render_template(
        "index.html",
        result=result,
        error=error,
        test_accuracy=f"{test_accuracy * 100:.2f}%",
        cat_f1=f"{report['Cat']['f1-score']:.4f}",
        dog_f1=f"{report['Dog']['f1-score']:.4f}",
        macro_f1=f"{report['macro avg']['f1-score']:.4f}",
        weighted_f1=f"{report['weighted avg']['f1-score']:.4f}",
        confusion_matrix=matrix,
        layer_rows=layer_rows,
        weight_rows=weight_rows
    )


@app.route("/<path:file_name>")
def project_file(file_name):
    allowed_files = {
        MODEL_DIAGRAM_FILE,
        "tensorflow_training_history.png",
        "tensorflow_confusion_matrix.png",
        "tensorflow_sample_prediction.png"
    }

    if file_name not in allowed_files:
        return "Not found", 404

    return send_from_directory(".", file_name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)

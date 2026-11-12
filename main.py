import os

import random
import cv2
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


IMAGE_SIZE = 128
TRAIN_RATIO = 0.70
RANDOM_STATE = 42
RUN_WEBCAM_AFTER_TRAINING = True
CAMERA_INDEX = 0


def prepare_dataset_folder():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    print("Dataset folder found successfully!")
    print("Dataset path:", DATASET_DIR)



def extract_features(image):
    """Same feature extraction is used for training, test and webcam."""
    if image is None:
        raise ValueError("Empty image received")

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()

   
    small_gray = cv2.resize(gray, (16, 16)).flatten() / 255.0

    return np.hstack([hog_features, hist, small_gray]).astype(np.float32)


def augment_image(image):
    
    h, w = image.shape[:2]
    augmented = [image]

    
    for angle in (-10, 10):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        augmented.append(rotated)

   
    augmented.append(cv2.convertScaleAbs(image, alpha=1.0, beta=20))
    augmented.append(cv2.convertScaleAbs(image, alpha=1.0, beta=-20))

   
    augmented.append(cv2.flip(image, 1))
    return augmented


def load_and_split_dataset():
    class_names = sorted(
        folder for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    )

    if len(class_names) < 2:
        raise ValueError("Expected class folders inside the dataset folder.")

    all_images, all_labels = [], []

    print("\nClasses and image counts:")
    for label, class_name in enumerate(class_names):
        folder = os.path.join(DATASET_DIR, class_name)
        image_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        valid_images = []
        for filename in image_files:
            path = os.path.join(folder, filename)
            image = cv2.imread(path)
            if image is not None:
                valid_images.append(image)

        if len(valid_images) < 2:
            raise ValueError(f"Not enough valid images in class: {class_name}")

        print(f"  {class_name}: {len(valid_images)} images")
        all_images.extend(valid_images)
        all_labels.extend([label] * len(valid_images))

   
    indices = np.arange(len(all_images))
    train_indices, test_indices = train_test_split(
        indices,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        stratify=np.array(all_labels)
    )

    X_train, y_train = [], []
    X_test, y_test = [], []

    print("\n70/30 split:")
    for label, class_name in enumerate(class_names):
        class_train = [i for i in train_indices if all_labels[i] == label]
        class_test = [i for i in test_indices if all_labels[i] == label]
        print(
            f"  {class_name:12s}: train={len(class_train):2d}, "
            f"test={len(class_test):2d}"
        )

   
    for index in train_indices:
        image = all_images[index]
        label = all_labels[index]
        for augmented_image in augment_image(image):
            X_train.append(extract_features(augmented_image))
            y_train.append(label)

 
    for index in test_indices:
        X_test.append(extract_features(all_images[index]))
        y_test.append(all_labels[index])

    return (
        np.asarray(X_train), np.asarray(y_train),
        np.asarray(X_test), np.asarray(y_test),
        class_names
    )



def train_and_evaluate(X_train, y_train, X_test, y_test, class_names):
    models = {
        "SVM": CalibratedClassifierCV(
    estimator=make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE
             )
                ),
    ensemble=False
              ),
       
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5, max_iter=4000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, weights="distance")
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=18,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Gaussian Naive Bayes": make_pipeline(
            StandardScaler(),
            GaussianNB()
        )
    }

    results = {}
    print("\n========== MODEL COMPARISON (30% TEST DATA) ==========")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
        results[name] = (model, accuracy, macro_f1)
        print(
            f"{name:22s} Accuracy: {accuracy * 100:6.2f}% | "
            f"Macro F1: {macro_f1 * 100:6.2f}%"
        )

    # Select using macro F1, not training accuracy.
    best_name = max(results, key=lambda name: results[name][2])
    best_model, best_accuracy, best_f1 = results[best_name]

    print(f"\nSelected model: {best_name}")
    print(f"Test accuracy: {best_accuracy * 100:.2f}%")
    print(f"Test macro F1: {best_f1 * 100:.2f}%")

    test_predictions = best_model.predict(X_test)
    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(
        y_test, test_predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    ))

  
    cm = confusion_matrix(
        y_test, test_predictions, labels=np.arange(len(class_names))
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        cm, display_labels=class_names
    ).plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name} (30% test)")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    scores = [results[n][1] * 100 for n in names]
    ax.bar(names, scores)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model comparison on untouched test data")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close(fig)

    
    model_data = {
        "model": best_model,
        "classes": class_names,
        "image_size": IMAGE_SIZE,
        "best_model_name": best_name,
        "test_accuracy": best_accuracy,
        "test_macro_f1": best_f1
    }
    model_path = os.path.join(BASE_DIR, "best_model.pkl")
    joblib.dump(model_data, model_path)
    print("\nSaved best_model.pkl, confusion_matrix.png, accuracy_comparison.png")
    return model_data



def predict_frame(frame, model_data):
    feature = extract_features(frame).reshape(1, -1)
    model = model_data["model"]
    classes = model_data["classes"]

    probabilities = model.predict_proba(feature)[0]
    order = np.argsort(probabilities)[::-1]
    best_index = int(order[0])
    confidence = float(probabilities[best_index])

   
    if confidence < 0.40:
        label = "UNKNOWN - move object closer"
    else:
        label = classes[best_index].upper()

    top_three = [
        (classes[int(i)], float(probabilities[int(i)]))
        for i in order[:3]
    ]
    return label, confidence, top_three



def run_webcam(model_data):
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Could not open webcam. Check camera permission/index.")
        camera.release()
        return

    print("\nWebcam started.")
    print("Place ONE object in the rectangle, against a plain background.")
    print("Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read from webcam.")
                break

            h, w = frame.shape[:2]

            # Calculate ROI dimensions
            roi_w = int(w * 0.72)
            roi_h = int(roi_w * 9 / 16)

            if roi_h > int(h * 0.72):
                roi_h = int(h * 0.72)
                roi_w = int(roi_h * 16 / 9)

            # Center the ROI
            x1 = (w - roi_w) // 2
            y1 = (h - roi_h) // 2
            x2, y2 = x1 + roi_w, y1 + roi_h

            roi = frame[y1:y2, x1:x2]

            # Predict object
            label, confidence, top_three = predict_frame(
                roi, model_data
            )

            # Draw ROI rectangle
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            # Display prediction
            cv2.putText(
                frame,
                f"{label} ({confidence * 100:.1f}%)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Display top three predictions
            for rank, (name, prob) in enumerate(top_three, start=1):
                text = f"{rank}. {name}: {prob * 100:.1f}%"

                cv2.putText(
                    frame,
                    text,
                    (20, 70 + rank * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )

            # Show webcam window
            cv2.imshow(
                "Object Classification - Press Q to quit",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print("Webcam error:", e)

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")



if __name__ == "__main__":
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    prepare_dataset_folder()
    X_train, y_train, X_test, y_test, class_names = load_and_split_dataset()

    print("\nFeature data:")
    print("Training samples after augmentation:", X_train.shape[0])
    print("Testing samples (original only):", X_test.shape[0])
    print("Number of classes:", len(class_names))

    trained_data = train_and_evaluate(
        X_train, y_train, X_test, y_test, class_names
    )

    if RUN_WEBCAM_AFTER_TRAINING:
        run_webcam(trained_data)
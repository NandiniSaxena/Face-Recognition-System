import os
import cv2
import numpy as np
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

def load_dataset(folder_path):
    images, labels, label_names = [], [], []
    label_map = {}

    for i, person_name in enumerate(sorted(os.listdir(folder_path))):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        label_map[i] = person_name
        for file in os.listdir(person_folder):
            file_path = os.path.join(person_folder, file)
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image)
                labels.append(i)
    return images, labels, label_map

def extract_encodings(images):
    encodings = []
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        if len(face_locations) == 1:
            encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
            encodings.append(encoding)
        else:
            encodings.append(None)
    return encodings

def filter_valid_encodings(encodings, labels):
    filtered_encodings, filtered_labels = [], []
    for enc, label in zip(encodings, labels):
        if enc is not None:
            filtered_encodings.append(enc)
            filtered_labels.append(label)
    return filtered_encodings, filtered_labels

def main():
    print("🔍 Loading training data...")
    train_images, train_labels, label_map = load_dataset("dataset/train")
    train_encodings = extract_encodings(train_images)
    train_encodings, train_labels = filter_valid_encodings(train_encodings, train_labels)
    print(f"✅ {len(train_encodings)} valid face encodings found for training.")

    if not train_encodings:
        print("❌ No valid face encodings found in training set.")
        return

    print("🤖 Training KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_encodings, train_labels)

    print("🧪 Loading test data...")
    test_images, test_labels, _ = load_dataset("dataset/test")
    test_encodings = extract_encodings(test_images)
    test_encodings, test_labels = filter_valid_encodings(test_encodings, test_labels)
    print(f"🧬 {len(test_encodings)} valid face encodings found for testing.")

    if not test_encodings:
        print("❌ No valid face encodings found in test set.")
        return

    print("🎯 Predicting and evaluating...")
    predictions = knn.predict(test_encodings)

    print("\n📊 Classification Report:")
    labels_present = unique_labels(test_labels, predictions)
    print(classification_report(
        test_labels, predictions,
        labels=labels_present,
        target_names=[label_map[i] for i in labels_present]
    ))

    print(f"🎯 Accuracy: {accuracy_score(test_labels, predictions) * 100:.2f}%")

if __name__ == "__main__":
    main()

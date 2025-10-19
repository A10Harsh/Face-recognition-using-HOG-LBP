import cv2
import numpy as np
import os
import glob
import itertools
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# --- 1. COPY ALL HELPER FUNCTIONS FROM YOUR STREAMLIT APP ---
# -----------------------------------------------------------------------------

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def apply_retinex(image):
    image = image.astype(np.float32) + 1.0
    gaussian = cv2.GaussianBlur(image, (101, 101), 30)
    retinex = np.log(image) - np.log(gaussian + 1.0)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(retinex)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_face(img):
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return img[y:y+h, x:x+w]

def extract_lbp_from_cropped_face(cropped, grid_x=8, grid_y=8):
    """
    This is your ENHANCED LBP function, copied directly.
    """
    if cropped is None:
        return None
    if cropped.ndim == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    radius, n_points, method = 1, 8, 'uniform'
    h, w = cropped.shape
    cell_h, cell_w = h // grid_y, w // grid_x
    all_histograms = []
    for i in range(grid_y):
        for j in range(grid_x):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = cropped[y1:y2, x1:x2]
            lbp = local_binary_pattern(cell, n_points, radius, method)
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            all_histograms.append(hist)
    final_hist = np.concatenate(all_histograms)
    final_hist = final_hist.astype("float32")
    final_hist /= (final_hist.sum() + 1e-6)
    return final_hist

class PrecomputedEmbedding(Embeddings):
    def embed_documents(self, texts): return []
    def embed_query(self, text): return []

def l2_to_similarity(distance, max_distance=2.0):
    """
    This is your custom similarity conversion, copied directly.
    """
    similarity = max(0.0, 1 - (distance / max_distance))
    similarity = (similarity - 0.999) * 1000
    return similarity

# -----------------------------------------------------------------------------
# --- 2. NEW BATCH PROCESSING & EVALUATION LOGIC ---
# -----------------------------------------------------------------------------

def histogram_from_image_path(img_path):
    """
    Processes an image from a file path and returns its LBP histogram.
    This function combines all your preprocessing steps.
    """
    image = cv2.imread(img_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_img = apply_clahe(gray)
    retinex_img = apply_retinex(clahe_img)
    cropped_face = detect_and_crop_face(retinex_img)
    hist = extract_lbp_from_cropped_face(cropped_face) # Using your enhanced LBP
    return hist

def load_dataset(dataset_path):
    """Loads a dataset from a structured directory (folder per person)."""
    face_dict = {}
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            image_paths = glob.glob(os.path.join(person_dir, '*.[jJ][pP][gG]')) + \
                          glob.glob(os.path.join(person_dir, '*.[pP][nN][gG]'))
            if image_paths:
                face_dict[person_name] = sorted(image_paths)
    return face_dict

def evaluate_identification(dataset, embedding_model):
    """Calculates Accuracy, Precision, Recall, F1 for 1:N Identification."""
    print("\n--- Starting 1:N Identification Evaluation ---")
    
    gallery_embeddings, gallery_metadatas = [], []
    probe_paths, probe_true_names = [], []

    print("Step 1: Building gallery (1st image) and probe (rest) sets...")
    for name, paths in tqdm(dataset.items()):
        if len(paths) < 2: continue
        embedding_vector = histogram_from_image_path(paths[0])
        if embedding_vector is not None:
            gallery_embeddings.append(("vector", embedding_vector.tolist()))
            gallery_metadatas.append({"name": name})
        for probe_path in paths[1:]:
            probe_paths.append(probe_path)
            probe_true_names.append(name)

    print("Step 2: Building FAISS index from gallery (using L2 distance)...")
    faiss_store = FAISS.from_embeddings(
        text_embeddings=gallery_embeddings,
        embedding=embedding_model,
        metadatas=gallery_metadatas
    )

    print(f"Step 3: Querying with {len(probe_paths)} probe images...")
    y_pred = []
    y_true = probe_true_names
    
    for i in tqdm(range(len(probe_paths))):
        query_vector = histogram_from_image_path(probe_paths[i])
        if query_vector is not None:
            results = faiss_store.similarity_search_with_score_by_vector(query_vector.tolist(), k=1)
            y_pred.append(results[0][0].metadata['name'] if results else "Unknown")
        else:
            y_pred.append("Face Not Detected")

    print("\n--- Identification Results ---")
    valid_indices = [i for i, p in enumerate(y_pred) if p not in ["Unknown", "Face Not Detected"]]
    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]
    
    if not y_true_filtered:
        print("Could not process any probe images. Cannot calculate metrics.")
        return

    print(f"✅ Overall Accuracy: {accuracy_score(y_true_filtered, y_pred_filtered):.2%}\n")
    print("📊 Classification Report:")
    print(classification_report(y_true_filtered, y_pred_filtered, zero_division=0))
    
    print("📊 Confusion Matrix:")
    labels = sorted(list(dataset.keys()))
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Identification Confusion Matrix')
    plt.show()

def evaluate_verification(dataset, threshold=0.82):
    """Calculates FAR and FNR for 1:1 Verification using your custom similarity."""
    print(f"\n--- Starting 1:1 Verification Evaluation (Threshold: {threshold}) ---")

    genuine_pairs = []
    imposter_pairs = []
    identities = list(dataset.keys())
    
    for i in range(len(identities)):
        person1_paths = dataset[identities[i]]
        if len(person1_paths) >= 2:
            genuine_pairs.extend(list(itertools.combinations(person1_paths, 2)))
        for j in range(i + 1, len(identities)):
            person2_paths = dataset[identities[j]]
            imposter_pairs.extend(list(itertools.product(person1_paths[:1], person2_paths[:1])))

    print(f"Generated {len(genuine_pairs)} genuine pairs and {len(imposter_pairs)} imposter pairs.")
    
    false_rejections = 0
    for path1, path2 in tqdm(genuine_pairs, desc="Testing Genuine Pairs"):
        vec1 = histogram_from_image_path(path1)
        vec2 = histogram_from_image_path(path2)
        if vec1 is not None and vec2 is not None:
            distance = np.linalg.norm(vec1 - vec2) # Calculate L2 Distance
            similarity = l2_to_similarity(distance) # Use your custom conversion
            if similarity < threshold:
                false_rejections += 1
    
    false_acceptances = 0
    for path1, path2 in tqdm(imposter_pairs, desc="Testing Imposter Pairs"):
        vec1 = histogram_from_image_path(path1)
        vec2 = histogram_from_image_path(path2)
        if vec1 is not None and vec2 is not None:
            distance = np.linalg.norm(vec1 - vec2) # Calculate L2 Distance
            similarity = l2_to_similarity(distance) # Use your custom conversion
            if similarity >= threshold:
                false_acceptances += 1

    fnr = false_rejections / len(genuine_pairs) if genuine_pairs else 0
    far = false_acceptances / len(imposter_pairs) if imposter_pairs else 0

    print("\n--- Verification Results ---")
    print(f"🔴 False Rejections (FR): {false_rejections} out of {len(genuine_pairs)}")
    print(f"🔴 False Acceptances (FA): {false_acceptances} out of {len(imposter_pairs)}")
    print(f"📈 False Rejection Rate (FNR): {fnr:.2%}")
    print(f"📈 False Acceptance Rate (FAR): {far:.2%}")

# -----------------------------------------------------------------------------
# --- 3. MAIN EXECUTION BLOCK ---
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # IMPORTANT: Set this path to your dataset folder.
    DATASET_PATH = "dataset" # <--- CONFIGURE THIS

    # This is the similarity threshold from your Streamlit app.
    # Experiment with this value to see how FAR and FNR change.
    SIMILARITY_THRESHOLD = 0.82 # <--- CONFIGURE THIS

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' does not exist.")
    else:
        dataset = load_dataset(DATASET_PATH)
        if len(dataset) < 2:
            print("Error: Dataset must contain at least two different people.")
        else:
            embedding_model = PrecomputedEmbedding()
            evaluate_identification(dataset, embedding_model)
            evaluate_verification(dataset, threshold=SIMILARITY_THRESHOLD)
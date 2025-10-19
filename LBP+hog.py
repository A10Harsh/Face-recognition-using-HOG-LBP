import streamlit as st
import cv2
import numpy as np
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from skimage.feature import local_binary_pattern, hog
from langchain.schema import Document
import uuid
import os
import shutil

# ----------------------------- HOG Feature Extraction -----------------------------
def extract_hog_features(image):
    if image is None:
        return None
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize for HOG (standard size)
    image_resized = cv2.resize(image, (128, 128))

    features = hog(image_resized,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   transform_sqrt=True,
                   feature_vector=True)
    return features

# ----------------------------- CLAHE -----------------------------
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# ----------------------------- Retinex -----------------------------
def apply_retinex(image):
    image = image.astype(np.float32) + 1.0
    gaussian = cv2.GaussianBlur(image, (101, 101), 30)
    retinex = np.log(image) - np.log(gaussian + 1.0)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(retinex)
    

# ----------------------------- Face Detection -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_face(img):
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return img[y:y+h, x:x+w]

# ----------------------------- ENHANCED LBP Extraction -----------------------------
def extract_lbp_from_cropped_face(cropped, grid_x=8, grid_y=8):
    if cropped is None:
        return None
    if cropped.ndim == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
    radius = 1
    n_points = 8 * radius
    method = 'uniform'
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

# ----------------------------- Custom Embedding Wrapper -----------------------------
class PrecomputedEmbedding(Embeddings):
    def embed_documents(self, texts):
        return []
    def embed_query(self, text):
        return []

# ----------------------------- Similarity Conversion -----------------------------
def l2_to_similarity(distance, max_distance=2.0):
    similarity = max(0.0, 1 - (distance / max_distance))
    # Adjust this scaling factor as needed for the new combined vectors
    print(similarity)
    return similarity * 100

# ✅ Utility function to get all stored metadata
def get_all_entries(faiss_store):
    entries = []
    if faiss_store and hasattr(faiss_store, "index_to_docstore_id"):
        for i, doc_id in faiss_store.index_to_docstore_id.items():
            try:
                doc = faiss_store.docstore.search(doc_id)
                if doc and "name" in doc.metadata and "id" in doc.metadata:
                    entries.append({"name": doc.metadata["name"], "id": doc.metadata["id"], "doc_id": doc_id})
            except Exception:
                pass
    return entries

# ----------------------------- Full Preprocessing Pipeline (CORRECTED) -----------------------------
# ----------------------------- Full Preprocessing Pipeline (CORRECTED) -----------------------------
def features_from_uploaded_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    clahe_img = apply_clahe(gray)
    retinex_img = apply_retinex(clahe_img)
    cropped_face = detect_and_crop_face(retinex_img)
    
    if cropped_face is None:
        return None
        
    lbp_hist = extract_lbp_from_cropped_face(cropped_face)
    hog_feat = extract_hog_features(cropped_face)

    if lbp_hist is not None and hog_feat is not None:
        # --- ADJUST THE WEIGHT HERE ---
        # A value of 1.0 gives equal importance.
        # A value > 1.0 gives LBP more importance.
        lbp_weight = 2.0# ✅ Make LBP twice as important
        
        
        # 1. Normalize LBP vector (as before)
        lbp_hist = lbp_hist.astype("float32")
        lbp_hist /= (np.linalg.norm(lbp_hist) + 1e-6)
        
        # 2. Normalize HOG vector (as before)
        hog_feat = hog_feat.astype("float32")
        hog_feat /= (np.linalg.norm(hog_feat) + 1e-6)

        # 3. ✅ APPLY THE WEIGHT TO THE LBP VECTOR
        weighted_lbp_hist = lbp_hist * lbp_weight
        
        # 4. Concatenate the weighted LBP and the original HOG vector
        combined_vector = np.concatenate([weighted_lbp_hist, hog_feat])
        return combined_vector
    else:
        return None

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="Face Biometric System (LBP+HOG)", layout="centered")
st.title("Face Biometric System (LBP + HOG)")

# Mode Selection
mode = st.radio("Select Mode:", ["Register New Face", "Identify Face (1:N)", "Delete Registered Face"])

embedding_model = PrecomputedEmbedding()
# ✅ FIX 3: Changed path to reflect new feature vectors
faiss_path = "combined_faiss_index" 

# Load existing FAISS index if available
if os.path.exists(faiss_path):
    try:
        faiss_store = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning(f"Error loading FAISS index: {e}. If you changed feature extraction methods, you must delete the old index folder.")
        faiss_store = None
else:
    faiss_store = None

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

# ----------------------------- Registration -----------------------------
if mode == "Register New Face":
    name = st.text_input("👤 Enter person's name")
    if uploaded_file and name:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        with st.spinner("Processing image and registering..."):
            feature_vector = features_from_uploaded_image(image)
            if feature_vector is not None:
                embedding_vector = feature_vector.tolist()
                person_id = str(uuid.uuid4())[:8]
                metadata = {"name": name, "id": person_id}
                doc = Document(page_content=f"Feature vector for {name}", metadata=metadata)

                if faiss_store is None:
                    faiss_store = FAISS.from_embeddings(
                        text_embeddings=[(doc.page_content, embedding_vector)],
                        embedding=embedding_model,
                        metadatas=[metadata]
                    )
                else:
                    faiss_store.add_embeddings(
                        text_embeddings=[(doc.page_content, embedding_vector)],
                        metadatas=[metadata]
                    )

                faiss_store.save_local(faiss_path)
                st.success(f"**{name}** registered successfully with ID: `{person_id}`")
            else:
                st.error("⚠️ No face detected. Please upload a clear face image.")

# ----------------------------- Identification (1:N) -----------------------------
elif mode == "Identify Face (1:N)":
    if faiss_store is None:
        st.error("No database found. Please register faces first.")
    elif uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing face and searching for matches..."):
            feature_vector = features_from_uploaded_image(image)
            if feature_vector is not None:
                embedding_vector = feature_vector.tolist()
                
                results_with_scores = faiss_store.similarity_search_with_score_by_vector(
                    embedding=embedding_vector, k=1
                )
                
                if results_with_scores:
                    doc, distance = results_with_scores[0]
                    similarity = l2_to_similarity(distance)

                    # NOTE: You will need to re-evaluate this threshold for the new LBP+HOG vectors
                    if similarity >= 55: 
                        st.subheader("✅ Match Found")
                        st.success(f"**Name:** {doc.metadata['name']}")
                        st.info(f"**ID:** {doc.metadata['id']}")
                        st.info(f"**Similarity Score:** {similarity:.2f}%")
                        print(similarity)
                    else:
                        print(similarity)
                        st.warning(f"No close match found. Closest is **{doc.metadata['name']}**.")
                        st.info(f"**Similarity Score:** {similarity:.2f}%")
                else:
            
                    st.warning("No entries found in the vector store to compare against.")
            else:
                st.error("No face detected in the uploaded image.")



# ----------------------------- Deletion -----------------------------
elif mode == "Delete Registered Face":
    if faiss_store is None:
        st.error(" No FAISS index found. Register faces first.")
    else:
        st.subheader(" Delete Person by ID")

        # --- NEW MESSAGE ---
        st.info("Enter the exact 8-character ID of the person you wish to delete. This action is irreversible.")
        # --- END NEW MESSAGE ---

        entries = get_all_entries(faiss_store)
        if not entries:
            st.warning("No entries found in the database.")
        else:
            # --- SECTION REMOVED AS REQUESTED ---
            # The list of registered persons is no longer displayed.
            # The user must know the ID to perform a deletion.
            
            del_id = st.text_input("Enter the full ID of the person to delete")

            if st.button("Delete Person"):
                if not del_id:
                    st.warning("Please enter an ID.")
                else:
                    doc_id_to_delete = None
                    for e in entries:
                        if e["id"] == del_id.strip():
                            doc_id_to_delete = e["doc_id"]
                            break
                    
                    if doc_id_to_delete:
                        try:
                            success = faiss_store.delete([doc_id_to_delete])
                            if success:
                                st.success(f"Successfully deleted entry with ID: `{del_id}`")
                                
                                # --- NEW POP-UP MESSAGE ---
                                st.toast(f"System: Person with ID {del_id} has been deleted.", icon="🗑️")
                                # --- END NEW POP-UP MESSAGE ---

                                if len(get_all_entries(faiss_store)) == 0:
                                    if os.path.exists(faiss_path):
                                        shutil.rmtree(faiss_path)
                                    st.info("Database is now empty. The index file has been removed.")
                                    faiss_store = None
                                else:
                                    faiss_store.save_local(faiss_path)
                                
                                # A small delay to let the toast appear before rerun
                                import time
                                time.sleep(1) 
                                st.rerun()

                            else:
                                st.error(" Deletion failed for an unknown reason.")

                        except Exception as e:
                            st.error(f"An error occurred during deletion: {e}")
                    else:
                        st.error(" ID not found. Please check the ID and try again.")
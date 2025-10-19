import streamlit as st
import cv2
import numpy as np
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from skimage.feature import local_binary_pattern
from langchain.schema import Document
import uuid
import os
import shutil

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
    """
    Extracting  a spatially enhanced LBP feature vector from a cropped face image.
    This function divides the image into a grid (e.g., 8x8) and computes an LBP
    histogram for each cell. It then concatenates these histograms into a single,
    powerful feature vector that preserves spatial information.
    """
    if cropped is None:
        return None

    # 1. Convert to grayscale if the image has color channels
    if cropped.ndim == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
    # LBP parameters
    radius = 1
    n_points = 8 * radius
    method = 'uniform'
    
    # Get image dimensions
    h, w = cropped.shape
    
    # Calculate cell dimensions
    cell_h, cell_w = h // grid_y, w // grid_x

    all_histograms = []
    
    # 2. Iterate over the grid
    for i in range(grid_y):
        for j in range(grid_x):
            # Define the boundaries of the current cell
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            
            # Extract the cell from the image
            cell = cropped[y1:y2, x1:x2]
            
            # 3. Compute LBP for the cell
            lbp = local_binary_pattern(cell, n_points, radius, method)
            
            # 4. Compute histogram for the LBP cell
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            
            all_histograms.append(hist)

    # 5. Concatenate all histograms into a single feature vector
    final_hist = np.concatenate(all_histograms)

    # 6. Normalize the final concatenated histogram (L1 norm)
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
    similarity = (similarity-.999)*1000
    return similarity

def verification_Normalization(distance, max_distance=2.0):
    similarity = max(0.0, 1 - (distance / max_distance))
    return similarity * 100

# ✅ Utility function to get all stored metadata
def get_all_entries(faiss_store):
    entries = []
    if faiss_store and hasattr(faiss_store, "index_to_docstore_id"):
        for doc_id in faiss_store.index_to_docstore_id.values():
            try:
                doc = faiss_store.docstore.search(doc_id)
                if doc and "name" in doc.metadata and "id" in doc.metadata:
                    entries.append({"name": doc.metadata["name"], "id": doc.metadata["id"], "doc_id": doc_id})
            except:
                pass
    return entries

# ----------------------------- Full Preprocessing Pipeline -----------------------------
def histogram_from_uploaded_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    clahe_img = apply_clahe(gray)
    retinex_img = apply_retinex(clahe_img)
    cropped_face = detect_and_crop_face(retinex_img)
    # This now calls the enhanced LBP function
    hist = extract_lbp_from_cropped_face(cropped_face)
    return hist

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="Face Skin Luminescence Biometric System", layout="centered")
st.title("Face Skin Luminescence Biometric")

# Mode Selection
mode = st.radio("Select Mode:", ["Register New Face", "Identify Face", "Verify Face (1:1)", "Delete Registered Face"])

embedding_model = PrecomputedEmbedding()
faiss_path = "lbp_faiss_index"

# Load existing FAISS index if available
if os.path.exists(faiss_path):
    try:
        faiss_store = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning(f"Error loading FAISS index: {e}")
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
            hist = histogram_from_uploaded_image(image)
            if hist is not None:
                embedding_vector = hist.tolist()
                person_id = str(uuid.uuid4())[:8]  # short unique ID
                metadata = {"name": name, "id": person_id}
                doc = Document(page_content=f"LBP vector for {name}", metadata=metadata)

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
elif mode == "Identify Face":
    if faiss_store is None:
        st.error("No FAISS database found. Please register some faces first.")
    elif uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing face and searching for matches..."):
            hist = histogram_from_uploaded_image(image)
            if hist is not None:
                embedding_vector = hist.tolist()
                
                results_with_scores = faiss_store.similarity_search_with_score_by_vector(
                    embedding=embedding_vector, k=1
                )
                
                if results_with_scores:
                    doc, distance = results_with_scores[0]
                    similarity = l2_to_similarity(distance)

                    # You may need to adjust this threshold based on the new feature vectors
                    if similarity >= 0.82: 
                        st.subheader("✅ Match Found")
                        st.success(f"**Name:** {doc.metadata['name']}")
                        st.info(f"**ID:** {doc.metadata['id']}")
                        st.info(f"**Similarity Score:** {similarity:.2f}")
                    else:
                        st.warning(f"No close match found. Closest is {doc.metadata['name']}")
                        st.info(f"**Similarity Score:** {similarity:.2f}")
                else:
                    st.warning("No similar image found in the vector store.")
            else:
                st.error("No face detected in the image.")

# ----------------------------- Verification (1:1) - NEW SECTION -----------------------------
elif mode == "Verify Face (1:1)":
    if faiss_store is None:
        st.error("No FAISS database found. Please register some faces first.")
    else:
        verify_id = st.text_input("👤 Enter the ID to verify against")
        
        if st.button("Verify Identity") and uploaded_file and verify_id:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image to Verify", use_column_width=True)

            with st.spinner("Processing face and verifying..."):
                # 1. Process the uploaded image to get its feature vector
                query_hist = histogram_from_uploaded_image(image)
                if query_hist is None:
                    st.error("No face detected in the uploaded image.")
                else:
                    query_vector = np.array(query_hist, dtype=np.float32)

                    # 2. Find the document and internal index for the provided ID
                    target_doc_info = None
                    for entry in get_all_entries(faiss_store):
                        if entry['id'] == verify_id.strip():
                            target_doc_info = entry
                            break
                    
                    if not target_doc_info:
                        st.error(f"ID `{verify_id}` not found in the database.")
                    else:
                        # 3. Reconstruct the stored vector from the FAISS index
                        docstore_id_to_index = {v: k for k, v in faiss_store.index_to_docstore_id.items()}
                        internal_faiss_index = docstore_id_to_index.get(target_doc_info['doc_id'])
                        
                        if internal_faiss_index is not None:
                            stored_vector = faiss_store.index.reconstruct(internal_faiss_index).flatten()
                            
                            # 4. Calculate L2 distance and similarity
                            distance = np.linalg.norm(query_vector - stored_vector)
                            similarity = verification_Normalization(distance)
                            
                            st.subheader("Verification Result")
                            # 5. Compare against threshold and show result
                            if similarity >= 0.82:
                                st.success(f"✅ Verified as **{target_doc_info['name']}**")
                                st.info(f"**ID:** {target_doc_info['id']}")
                                st.info(f"**Similarity Score:** {similarity:.2f}")
                            else:
                                st.error(f"❌ Not Verified. Does not match **{target_doc_info['name']}**.")
                                st.info(f"**ID Compared Against:** {target_doc_info['id']}")
                                st.info(f"**Similarity Score:** {similarity:.2f}")
                        else:
                            st.error("Could not locate the vector for the given ID in the FAISS index.")

# ----------------------------- Deletion -----------------------------
elif mode == "Delete Registered Face":
    if faiss_store is None:
        st.error("No FAISS index found. Register faces first.")
    else:
        st.subheader("Delete Person by ID")

        entries = get_all_entries(faiss_store)
        if not entries:
            st.warning("No entries found in the database.")
        else:
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
                                
                                # If the database is now empty, remove the folder
                                if len(get_all_entries(faiss_store)) == 0:
                                    if os.path.exists(faiss_path):
                                        shutil.rmtree(faiss_path)
                                    st.info("Database is now empty. The index file has been removed.")
                                    faiss_store = None
                                else:
                                    # Save the modified index
                                    faiss_store.save_local(faiss_path)

                                st.rerun()

                            else:
                                st.error("Deletion failed for an unknown reason.")

                        except Exception as e:
                            st.error(f"An error occurred during deletion: {e}")
                    else:
                        st.error("ID not found. Please check the ID and try again.")
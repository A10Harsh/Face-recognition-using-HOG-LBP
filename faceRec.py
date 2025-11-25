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

# ----------------------------- LBP -----------------------------
def extract_lbp_from_cropped_face(cropped):
    if cropped is None:
        return None
    if len(cropped.shape) == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    radius = 1
    n_points = 8 * radius
    method = 'uniform'
    lbp = local_binary_pattern(cropped, n_points, radius, method)

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# ----------------------------- Custom Embedding -----------------------------
class PrecomputedEmbedding(Embeddings):
    def embed_documents(self, texts):
        return []
    def embed_query(self, text):
        return embedding_vector

# ----------------------------- Similarity -----------------------------
def l2_to_similarity(distance, max_distance=4.0):
    return max(0.0, 1 - (distance / max_distance))

# ----------------------------- Image to LBP Vector -----------------------------
def histogram_from_uploaded_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    clahe_img = apply_clahe(gray)
    retinex_img = apply_retinex(clahe_img)
    cropped_face = detect_and_crop_face(retinex_img)
    hist = extract_lbp_from_cropped_face(cropped_face)
    return hist

# ----------------------------- Streamlit App -----------------------------
st.set_page_config(page_title="Face Matcher with LBP + FAISS", layout="centered")
st.title(" Face Image Similarity Search")
st.write("Upload a face image to find the most similar entry in the vector store.")

# Load FAISS vector store
embedding_vector = [0] * 59  # placeholder just to initialize the embedding model
embedding_model = PrecomputedEmbedding()

try:
    faiss_store = FAISS.load_local("lbp_faiss_index", embedding_model, allow_dangerous_deserialization=True)
    st.success(" FAISS index loaded successfully.")
except Exception as e:
    st.error(f" Failed to load FAISS index: {e}")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)


    with st.spinner("Processing and searching..."):
        hist = histogram_from_uploaded_image(image)
        if hist is not None:
            embedding_vector = hist.tolist()
            results_with_scores = faiss_store.similarity_search_with_score(embedding_vector, k=1)

            if results_with_scores:
                doc, distance = results_with_scores[0]
                similarity = l2_to_similarity(distance)

                st.subheader(" Match Found")
                st.write(f"**Similarity:** {similarity:.2f}")
                st.write(f"**Metadata:**")
                st.json(doc.metadata)
            else:
                st.warning(" No similar image found in the vector store.")
        else:
            st.error(" No face detected in the image.")



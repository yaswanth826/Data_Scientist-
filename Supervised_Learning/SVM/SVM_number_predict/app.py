"""
Simple API-style Streamlit app for digit recognition.
"""
import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Page configuration
st.set_page_config(page_title="Digit Recognizer API", page_icon="🔢")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_package = joblib.load('digit_model.pkl')
        return model_package
    except FileNotFoundError:
        st.error("Model file 'digit_model.pkl' not found. Please run 'train_model.py' first.")
        return None

def preprocess_image(image):
    """
    Preprocess the uploaded image to match the model's expected input.
    sklearn digits dataset has: dark background (0), light digits (higher values)
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 8x8 (same as sklearn digits dataset)
    image = image.resize((8, 8))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Invert based on mean value: if mean > 128, background is light, digit is dark
    if np.mean(img_array) > 128:
        # Light background, dark digit - invert to get dark background, light digit
        img_array = 255 - img_array
    
    # Apply threshold to clean up the image
    img_array = np.where(img_array < 64, 0, img_array)
    img_array = np.where(img_array > 192, 255, img_array)
    
    # Flatten and normalize to 0-16 range (sklearn digits range)
    img_flat = img_array.flatten().astype(float)
    img_normalized = img_flat / 255.0 * 16.0
    
    return img_normalized.reshape(1, -1)

def main():
    # Load model
    model_package = load_model()
    
    if model_package is None:
        return
    
    model = model_package['model']
    
    # Simple form-style interface
    st.title("🔢 Digit Recognition API")
    
    with st.form("digit_form"):
        uploaded_file = st.file_uploader("Upload digit image", type=['png', 'jpg', 'jpeg'])
        submit = st.form_submit_button("Recognize Digit")
    
    if submit and uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=200)
        
        # Preprocess and predict
        try:
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0]
            st.success(f"## Predicted Digit: {prediction}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    elif submit and uploaded_file is None:
        st.warning("Please upload an image first")

if __name__ == "__main__":
    main()


import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import os

# Load the trained image model
image_model = tf.keras.models.load_model("E:\CAPSTONE PROJECT\landing page\skin.keras")
model = tf.keras.models.load_model('models.keras')
data = pd.read_csv('training.csv')

# Mapping for sorethroat column
sorethroat_map = {'Yes': 1, 'No': 0}
data_cat = ['normal','measles']
# Landing Page
def landing_page():
    st.title("Rubeola and Rubella Detecting System (RRDS)")
    st.write("Welcome to RRDS, your reliable tool for accurate diagnosis of Rubeola and Rubella!")
    st.write("The RRDS is designed to assist healthcare professionals in obtaining precise diagnostic results for these infectious diseases.")
    st.write("Key Features:")
    st.write("- Accurate diagnostic results for Rubeola and Rubella")
    st.write("- Accessible from any healthcare location with electricity and internet")
    st.write("- Utilizes machine learning and complex algorithms trained on diverse data from various hospitals")
    st.write("Ready to get started? Click the button below!")

    # Start Button
    if st.button("Let's Start"):
        st.session_state.page = "Upload Image"
img_height = 180
img_width = 180
# Upload Image Page
def upload_image():
    st.title("Upload Image for Diagnosis")
    st.write("Upload an image of a suspected Rubeola or Rubella case for diagnosis.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Image Upload Logic and Prediction
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_image_path = "temp_image.png"  # Temporary path, change as needed
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and preprocess the image
        image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_bat = np.expand_dims(img_arr, axis=0)

        # Predict the image class
        predict = image_model.predict(img_bat)
        score = tf.nn.softmax(predict)
        predicted_class = data_cat[np.argmax(score)]

        # Display the image and prediction result
        st.image(image_load, width=200)
        st.write('Skin in image is ' + predicted_class)
        st.write('With accuracy of ' + str(np.max(score) * 100))

        # Collect additional information if diagnosis is measles
        if predicted_class == "measles":
            incubation = st.number_input('Incubation period (days)', min_value=0, max_value=20, value=10)
            sorethroat = st.selectbox('Sore Throat', ('Yes', 'No'))
            temperature = st.number_input('Temperature (°C)', min_value=30.0, max_value=45.0, value=36.5)
            sorethroat_encoded = sorethroat_map[sorethroat]
            input_data = pd.DataFrame({
                'incubation': [incubation],
                'sorethroat': [sorethroat_encoded],
                'temperature': [temperature]
            })
            if st.button("Submit"):
                prediction = model.predict(input_data)
                predicted_class = "Measles" if prediction[0] > 0.5 else "German Measles"
                # Display the result
                st.header("Diagnosis Prediction")
                st.write(f"The predicted diagnosis is: **{predicted_class}**")
        else:
            st.write(f"The skin is out of scope in the study thank you")
        # Delete the temporary image file
        os.remove(temp_image_path)

    # Back Button
    if st.button("Back"):
        st.session_state.page = "Landing Page"


# Main function
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Landing Page"

    # Page navigation
    if st.session_state.page == "Landing Page":
        landing_page()
    elif st.session_state.page == "Upload Image":
        upload_image()

# Run the app
if __name__ == "__main__":
    main()

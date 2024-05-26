import tensorflow as tf
import streamlit as st
import numpy as np
import os

# Function to get additional information about the detected plant species
def get_plant_info(plant_name):
    plant_info = {
        'banana': {
            'Common Name': 'Banana',
            'Scientific Name': 'Musa',
            'Habitat': 'Tropical regions',
            'Characteristics': 'Long, curved fruit with yellow skin and soft, sweet flesh',
            'More Info': 'https://en.wikipedia.org/wiki/Banana',
            'YouTube Video': 'pSsDEewm-ow'
        },
        'corn': {
            'Common Name': 'Corn',
            'Scientific Name': 'Zea mays',
            'Habitat': 'Temperate regions',
            'Characteristics': 'Grain plant producing large ears with kernels',
            'More Info': 'https://en.wikipedia.org/wiki/Maize',
            'YouTube Video': 'SNLypjH3cgA'
        },
        'eggplant': {
            'Common Name': 'Eggplant',
            'Scientific Name': 'Solanum melongena',
            'Habitat': 'Warm climates',
            'Characteristics': 'Purple, white, or green fruit with spongy flesh',
            'More Info': 'https://en.wikipedia.org/wiki/Eggplant',
            'YouTube Video': 'hU8cBG1l3sg'
        },
        'mango': {
            'Common Name': 'Mango',
            'Scientific Name': 'Mangifera indica',
            'Habitat': 'Tropical regions',
            'Characteristics': 'Sweet, juicy fruit with a thick skin',
            'More Info': 'https://en.wikipedia.org/wiki/Mango',
            'YouTube Video': 'SJfOARBwREU'
        },
        'watermelon': {
            'Common Name': 'Watermelon',
            'Scientific Name': 'Citrullus lanatus',
            'Habitat': 'Warm climates',
            'Characteristics': 'Large, round fruit with green rind and juicy, red flesh',
            'More Info': 'https://en.wikipedia.org/wiki/Watermelon',
            'YouTube Video': 'SgaEhH-8cMo'
        }
    }
    return plant_info.get(plant_name, {})
st.set_page_config(page_title="Plant Identification App", page_icon="🌿")


# Page title and objectives
st.title("Plant Identification System Enhances Biodiversity Conservation Effort")
st.write("The primary goal of this project was to develop an automated system using image recognition technology that can reliably identify different plant species in real time. In order to help academics, environmentalists, and conservationists make more informed judgments and support the preservation of biodiversity, the system was designed to make the process of gathering data easier.")

# Load the pre-trained model
model = tf.keras.models.load_model("E:/CAPSTONE PROJECT/planting/image_v2.keras")

# Define data categories and folders
data_cat = ['banana', 'corn', 'eggplant', 'mango', 'watermelon']
data_folder = {
    'None': None,
    'banana': 'banana',
    'corn': 'corn',
    'eggplant': 'eggplant',
    'mango': 'mango',
    'watermelon': 'watermelon',
    'other': 'other'
}
img_height = 180
img_width = 180

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Load and preprocess the image
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = np.expand_dims(img_arr, axis=0)

    # Predict the image class
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    predicted_class = data_cat[np.argmax(score)]

    # Get additional information about the detected plant species
    plant_info = get_plant_info(predicted_class)

    # Display the image and prediction result
    st.image(image_load, width=200)
    st.write('Veg/Fruit in image is ' + predicted_class)
    st.write('With accuracy of ' + str(np.max(score) * 100))

    # Display additional information
    st.subheader("Plant Information:")
    for key, value in plant_info.items():
        if key != "YouTube Video":
            st.write(f"- **{key}:** {value}")

    # Embed YouTube video
    if 'YouTube Video' in plant_info:
        st.video(f"https://www.youtube.com/watch?v={plant_info['YouTube Video']}")

    # Option to correct classification
    option = st.radio("Is the classification accurate?", ("", "No"))

    if option == "No":
        # Select folder to save the misclassified image
        folder = st.selectbox("Select the folder to save this image for training:", list(data_folder.keys()))
        if folder != 'None':
            save_path = os.path.join("E:/CAPSTONE PROJECT/planting/training_data", data_folder[folder])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.write(f"Image saved in {save_path}.")

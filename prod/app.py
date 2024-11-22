import streamlit as st
from PIL import Image
import random
import os
from utils import get_image, transform_to_night

image_pool = {
    "Buenos Aires": "data/data_test/buenos_aires.jpg",
    "San rafael": "data/data_test/sanra.jpg",
    "Mendoza": "data/data_test/mza.jpg",
    "Calle": "data/data_test/street.jpg",
}

# List of images to pick randomly later
test_dataset = ["data/day_val/" + path for path in os.listdir("data/day_val")]


st.title("Day-to-Night")
st.subheader("Elegi una imagen y transformala a su versión nocturna")

# Select model
model_name = st.selectbox(
    "Elegí el modelo que transformará tu imagen:", ["unet50", "unet50_basic", "unet18"]
)


# Option to select image from pool or upload
option = st.radio(
    "Imagen de entrada:",
    [
        "Elegir de un pool prediseñado",
        "Subir tu propia imagen",
        "Random de dataset de test",
    ],
)

if option == "Elegir de un pool prediseñado":
    selected_image = st.selectbox("Selecciona una imagen:", list(image_pool.keys()))
    image_path = image_pool[selected_image]
    image = Image.open(image_path)
    st.image(
        get_image(image), caption=f"Original - {selected_image}", use_column_width=True
    )

elif option == "Subir tu propia imagen":
    uploaded_file = st.file_uploader("Subí una imagen", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            get_image(image), caption="Original - Imagen subida", use_column_width=True
        )

elif option == "Random de dataset de test":
    if test_dataset:
        random_image_path = random.choice(test_dataset)
        image = Image.open(random_image_path)
        st.image(
            get_image(image),
            caption=f"Original - Random del dataset de testo",
            use_column_width=True,
        )
    else:
        st.warning("El dataset de testeo esta vacío o no existe.")

if "image" in locals():
    st.write("Imagen de noche:")
    night_image = transform_to_night(image, model_name)
    st.image(
        night_image,
        caption=f"Imagen transformada (Model: {model_name})",
        use_column_width=True,
    )

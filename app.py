# - Importing the dependencies
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import easyocr
import io

# - CSS Styling
st.markdown(
    """
    <style>
    .centered-heading {
        text-align: center;
        padding-bottom: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# - Defining a function to load the YOLO model
@st.cache_data()
def load_model():
    return YOLO('best.pt')

# - Defining a function to initialize the EasyOCR reader
@st.cache_data()
def initialize_easyocr():
    return easyocr.Reader(['en'])

# - Loading the pre-trained YOLOv8l model
model = load_model()

# - Initialize EasyOCR reader
reader = initialize_easyocr()

st.markdown("<h1 class='centered-heading'>Vehicle Registration Plate Detection App (YoloV8)</h1>", unsafe_allow_html=True)

# - Correction dictionary 1 for 1st character of province name
correction_dict_1 = {
    'H': 'W',
    'V': 'W',
    'M': 'W',
    'Y': 'W',
    'K': 'W'
}

# - Correction dictionary 2 for 1st character of province name
correction_dict_2 = {
    'H': 'W'
}

# - List of valid provinces
valid_provinces = ['CP', 'EP', 'NC', 'NE', 'NW', 'SB', 'SP', 'UP', 'WP']


def process_vehicle_registration_plate(cropped_image, reader):
    plate_image = cropped_image
    plate_image_np = np.array(plate_image)

    # - Converting the image to grayscale
    gray_image = cv2.cvtColor(plate_image_np, cv2.COLOR_BGR2GRAY)

    # - Getting the width and height of the gray_image
    total_height, total_width = gray_image.shape

    # - Calculating the width of the split
    split_width_0 = int(0.02 * total_width)  
    split_width_1 = int(0.20 * total_width)
    split_width_2 = int(0.18 * total_width)

    # - Calculating the height of the split
    split_height_1 = int(0.40 * total_height)

    # - Splitting the gray_image into two parts
    main_split_1 = gray_image[:, split_width_0:split_width_1]
    main_split_2 = gray_image[:, split_width_2:]

    # - Splitting the main_split_1 into two parts
    sub_split_1 = main_split_1[split_height_1:,:]

    # - Getting the width and height of the main_split_2
    part_2_height, part_2_width = main_split_2.shape

    # - Calculating the width of the split
    split_width_3 = int(0.45 * part_2_width) 

    # - Splitting the main_split_2 into two parts
    sub_split_2 = main_split_2[:, :split_width_3]

    # - Calculating the width of the split
    split_width_4 = int(0.45 * part_2_width)  # 16% of the total width
    split_width_5 = int(0.98 * part_2_width)  # 16% of the total width

    # Splitting the main_split_2 into two parts
    sub_split_3 = main_split_2[:, split_width_4:split_width_5]

    # Applying thresholding to segment characters from the background
    _, binary_image1 = cv2.threshold(sub_split_1, 224, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_image2 = cv2.threshold(sub_split_2, 224, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_image3 = cv2.threshold(sub_split_3, 224, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # - Character recognition on the preprocessed image
    results1 = reader.readtext(binary_image1)
    results2 = reader.readtext(binary_image2)
    results3 = reader.readtext(binary_image3)
    
    return results1, results2, results3


def process_text_results(results1, results2, results3, correction_dict_1, correction_dict_2, valid_provinces):
    # - Extracting the text from results1
    corrected_text1 = ""
    for _, text, _ in results1:
        text = text.replace(' ', '')
        if len(text) == 2:
            text = text.upper()
            first_letter = text[0]
            second_letter = text[1]

            # Checking if first_letter needs correction
            if first_letter in correction_dict_1:
                corrected_first_letter = correction_dict_1[first_letter]
            else:
                corrected_first_letter = first_letter

            # Checking if second_letter needs correction
            if second_letter in correction_dict_2:
                corrected_second_letter = correction_dict_2[second_letter]
            else:
                corrected_second_letter = second_letter

            # Building the corrected text
            corrected_text1 += corrected_first_letter + corrected_second_letter

            if corrected_text1 not in valid_provinces:
                corrected_text1 = ''

        else:
            corrected_text1 = ''

    # Extracting the text from results2
    corrected_text2 = ""
    for _, text, _ in results2:
        text = text.replace(' ', '')
        corrected_text2 = text.upper()
        if len(corrected_text2) != 3:
            corrected_text2 = ''

    # Extracting the text from results3
    corrected_text3 = ""
    for _, text, _ in results3:
        text = text.replace(' ', '')
        if len(text) > 4:
            corrected_text3 = text[:4]
        else:
            corrected_text3 = text
            
    return corrected_text1, corrected_text2, corrected_text3

# - Uploading multiple images
with st.form("my-form", clear_on_submit=True):
        uploaded_images = st.file_uploader("Upload an image or multiple images for vehicle registration plate detection", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("submit")

if uploaded_images:
    for uploaded_image in uploaded_images:
        # - Converting the uploaded image to a format compatible with YOLO
        img_bytes = uploaded_image.read()
        image = Image.open(io.BytesIO(img_bytes))

        results = model(source=image)

        for result in results:
            boxes = result.boxes
            
        number_of_plates = boxes.shape[0]

        # - Displaying the detection results for each image
        for result in results:
            im_array = result.plot()  
            im = Image.fromarray(im_array[..., ::-1])

            # - Displaying the image with bounding boxes
            st.image(im, caption=f"Image {uploaded_images.index(uploaded_image) + 1}. Object Detection Result - {number_of_plates} vehicle registration plates are detected.", use_column_width=True)
            
            for box in boxes:
                cord = box
                
                # - Extracting values from the bounding box
                x_center, y_center, box_width, box_height = cord.xywh[0]
                
                # - Calculating the top-left and bottom-right coordinates of the bounding box
                x1 = int(x_center - (box_width / 2))
                y1 = int(y_center - (box_height / 2))
                x2 = int(x_center + (box_width / 2))
                y2 = int(y_center + (box_height / 2))
                
                # - Converting the JpegImageFile object to a NumPy array
                image_np = np.array(image)
                
                # - Cropping the image using the bounding box coordinates
                cropped_image = image_np[y1:y2, x1:x2]
                cropped_image = Image.fromarray(cropped_image)
                
                # - Displaying the cropped image
                st.image(cropped_image, caption='Vehicle Registration Plate', use_column_width=True)
                
                # - Results of the EasyOCR
                results1, results2, results3 = process_vehicle_registration_plate(cropped_image, reader)
                
                # - Correcting the recognized text
                corrected_text1, corrected_text2, corrected_text3 = process_text_results(results1, results2, results3, correction_dict_1, correction_dict_2, valid_provinces)

                # - Displaying the Recognized Number Plate
                st.write(f'Recognized Number Plate: {corrected_text1} {corrected_text2} {corrected_text3}')
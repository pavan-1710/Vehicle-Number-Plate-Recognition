import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
import easyocr
import imutils
from PIL import Image
import csv
import os
import pandas as pd
import re

# -------------------------------
# Configs and CSV Initialization
# -------------------------------
CSV_FILE = "apartment_car_data.csv"
FIELDS = ["building_no", "door_no", "name", "contact_no", "car_color", "car_numberplate", "car_status"]

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()

# --- License Plate Detection Functions ---
def preprocess_image_for_plate(image):
    """Preprocess the image to detect the license plate."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    return gray, edged

def find_license_plate_contour(edged):
    """Find the contour of the license plate."""
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    return location

def correct_license_plate_format(text):
    """
    Correct common OCR errors in license plates and format them properly.
    Based on Indian license plate format: [STATE CODE] [DISTRICT NUMBER] [LETTER SERIES] [NUMBER]
    e.g., MH 14 DT 8831
    """
    # Convert to uppercase
    text = text.upper()
    
    # Common OCR mistake corrections
    text = text.replace('Z', '2')
    text = text.replace('6J', 'GJ')
    
    # Special corrections for patterns like KA33658899 or KA33G58899
    if re.match(r'[A-Z]{2}\d{2}[6G]\d5\d{4}', text):
        # Change 6 to G and 5 to S
        text = text.replace('6', 'G')
        # Find the 5 that appears in the letter section (not in the final numbers)
        match = re.search(r'[A-Z]{2}\d{2}[G](\d5)\d{4}', text)
        if match:
            letter_part = match.group(1)
            new_letter_part = letter_part.replace('5', 'S')
            text = text.replace(letter_part, new_letter_part)
    else:
        # More general corrections
        text = text.replace('6', 'G')
        
        # Replace L5 with 45
        text = text.replace('L5', '45')
        
        # Specific fix for common series codes
        text = text.replace('G5', 'GS')  # Like "G5" should be "GS"
        
        # In letter section, 5 is often misrecognized S
        # We need to identify the letter section
        parts = text.split()
        if len(parts) >= 3:  # We have state, district, and letter parts
            # Check if third part has a 5 in it
            if '5' in parts[2] and len(parts[2]) <= 2:  # Typical letter part is 2 chars
                parts[2] = parts[2].replace('5', 'S')
                text = ' '.join(parts)
    
    # Special fix for MH O2 FE cases
    text = text.replace('O2', 'O2')
    
    # Add spaces for standard license plate format if not present
    # Looking for patterns like: MH14DT8831 → MH 14 DT 8831
    if len(text) > 6 and ' ' not in text:
        # Typical format for Indian plates
        if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$', text):
            # Find where the first number section starts
            match = re.search(r'[A-Z]{2}(\d{1,2})', text)
            if match:
                start_idx = match.start(1)
                # Insert space after state code
                text = text[:start_idx] + ' ' + text[start_idx:]
                
                # Find where the letter series starts after district number
                match = re.search(r'\d{1,2}([A-Z]{1,2})', text)
                if match:
                    start_idx = match.start(1)
                    # Insert space before letter series
                    text = text[:start_idx] + ' ' + text[start_idx:]
                    
                    # Find where the final number starts
                    match = re.search(r'[A-Z]{1,2}(\d{1,4})', text)
                    if match:
                        start_idx = match.start(1)
                        # Insert space before final number
                        text = text[:start_idx] + ' ' + text[start_idx:]
    
    # NEW CODE: Fix letters in the last 4 characters (which should be numbers)
    parts = text.split()
    if parts:
        last_part = parts[-1]
        # Check if this is likely the number part (last segment)
        if len(last_part) <= 4:
            # Convert common letter substitutions in the last segment (should be all digits)
            new_last_part = ""
            for char in last_part:
                if char == 'G':
                    new_last_part += '6'
                elif char == 'S':
                    new_last_part += '5'
                elif char == 'O':
                    new_last_part += '0'
                elif char == 'I':
                    new_last_part += '1'
                elif char == 'B':
                    new_last_part += '8'
                else:
                    new_last_part += char
            parts[-1] = new_last_part
            text = ' '.join(parts)
    
    return text

def extract_plate_text(image, gray, location):
    """Extract text from the license plate."""
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)
    
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    
    if len(result) >= 2:
        text = result[0][-2] + " " + result[1][-2]
    else:
        text = "Not detected" if not result else result[0][-2]
    
    # Apply corrections to the detected text
    corrected_text = correct_license_plate_format(text)
    
    return corrected_text, cropped_image, location

# --- Car Color Detection Functions ---
def extract_dominant_colors(image, k=3):
    """Extract the dominant colors from an image using K-means clustering."""
    pixels = image.reshape(-1, 3)
    clt = KMeans(n_clusters=k)
    clt.fit(pixels)
    hist = np.bincount(clt.labels_)
    sorted_indices = np.argsort(hist)[::-1]
    sorted_colors = clt.cluster_centers_[sorted_indices]
    sorted_percentages = hist[sorted_indices] / hist.sum()
    sorted_colors = sorted_colors.astype(int)
    return sorted_colors, sorted_percentages

def process_image_for_color_detection(image):
    """Process an image to identify the color of a car."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    roi_x = width // 4
    roi_y = height // 3
    roi_width = width // 2
    roi_height = height // 3
    
    roi_rgb = image_rgb[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    dominant_colors, color_percentages = extract_dominant_colors(roi_rgb, k=3)
    dominant_color_rgb = dominant_colors[0]
    
    pixel = np.uint8([[dominant_color_rgb]])
    dominant_color_hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0]
    color_name = identify_color_from_hsv(dominant_color_hsv)
    
    return {
        'image_rgb': image_rgb,
        'roi_rgb': roi_rgb,
        'roi_coordinates': (roi_x, roi_y, roi_width, roi_height),
        'dominant_color_rgb': dominant_color_rgb,
        'dominant_color_hsv': dominant_color_hsv,
        'color_percentages': color_percentages,
        'all_dominant_colors': dominant_colors,
        'color_name': color_name
    }

def identify_color_from_hsv(hsv_color):
    """Identify the color name based on HSV values."""
    h, s, v = hsv_color
    h_normalized = h / 180.0
    s_normalized = s / 255.0
    v_normalized = v / 255.0
    
    if s_normalized < 0.15 and v_normalized > 0.8:
        return "white"
    elif s_normalized < 0.15 and v_normalized < 0.3:
        return "black"
    elif s_normalized < 0.15:
        return "gray"
    elif (h_normalized < 0.05 or h_normalized > 0.95) and s_normalized > 0.4:
        return "red"
    elif 0.05 <= h_normalized < 0.10 and s_normalized > 0.4:
        return "orange"
    elif 0.10 <= h_normalized < 0.17 and s_normalized > 0.4:
        return "yellow"
    elif 0.17 <= h_normalized < 0.33 and s_normalized > 0.4:
        return "green"
    elif 0.33 <= h_normalized < 0.52 and s_normalized > 0.4:
        return "cyan"
    elif 0.52 <= h_normalized < 0.74 and s_normalized > 0.4:
        return "blue"
    elif 0.74 <= h_normalized < 0.82 and s_normalized > 0.4:
        return "purple"
    elif 0.82 <= h_normalized < 0.95 and s_normalized > 0.4:
        return "magenta"
    else:
        return "unknown"

# --- Combined Processing Function ---
def process_image(image):
    """Process the image to detect car color and number plate."""
    # License plate detection
    gray, edged = preprocess_image_for_plate(image)
    location = find_license_plate_contour(edged)
    plate_text = "Not detected"
    if location is not None:
        plate_text, _, _ = extract_plate_text(image, gray, location)
    
    # Car color detection
    color_data = process_image_for_color_detection(image)
    color_name = color_data['color_name']
    
    return plate_text, color_name

# Function to check if license plate exists and get resident info with status
def get_plate_info(plate_number):
    """
    Check if the license plate exists in the CSV file and return details.
    Returns (exists, resident_data, current_status)
    """
    if not os.path.exists(CSV_FILE):
        return False, None, None
    
    try:
        df = pd.read_csv(CSV_FILE)
        matching_rows = df[df["car_numberplate"] == plate_number]
        
        if not matching_rows.empty:
            # Return the most recent entry (last one)
            resident_data = matching_rows.iloc[-1].to_dict()
            current_status = resident_data["car_status"]
            return True, resident_data, current_status
        return False, None, None
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return False, None, None

# -------------------------------
# Streamlit App Interface
# -------------------------------
def main():
    st.set_page_config(page_title="Vehicle Number Plate Recognition System", layout="centered")
    st.title("📹 Vehicle Number Plate Recognition System")

    init_csv()

    uploaded_file = st.file_uploader("📤 Upload a Car Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        bgr_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Process image
        number_plate, color_name = process_image(bgr_img)

        st.image(image, caption="Uploaded Car Image", use_container_width=True)
        st.success(f"🟥 Detected Car Color: {color_name}")
        st.success(f"🔤 Detected Number Plate: {number_plate}")

        # Check if the plate already exists in our records and get its status
        plate_exists, resident_data, current_status = get_plate_info(number_plate)

        if plate_exists:
            # Toggle the status based on the current status
            if current_status == "in":
                car_status = "out"
                status_msg = "Exiting"
            else:  # current_status is "out"
                car_status = "in"
                status_msg = "Entering"
                
            st.info(f"🚗 Car status: {status_msg} the apartment")
            st.success("✅ Car recognized! Details retrieved from database.")
            
            # Display the retrieved details
            st.write("### 🔍 Resident Details")
            details_to_show = {
                "Building No": resident_data["building_no"],
                "Door No": resident_data["door_no"],
                "Name": resident_data["name"],
                "Contact No": resident_data["contact_no"],
                "Car Color": color_name,
                "Number Plate": number_plate,
                "Status": car_status
            }
            st.write(details_to_show)
            
            # Automatically save the record with toggled status
            with open(CSV_FILE, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writerow({
                    "building_no": resident_data["building_no"],
                    "door_no": resident_data["door_no"],
                    "name": resident_data["name"],
                    "contact_no": resident_data["contact_no"],
                    "car_color": color_name,
                    "car_numberplate": number_plate,
                    "car_status": car_status
                })
            
            st.success(f"✅ {status_msg} record saved automatically!")
            
        else:
            # New car is entering - ask for details
            car_status = "in"
            st.info("🚗 Car status: Entering the apartment")
            st.warning("⚠️ New car detected! Please enter resident details.")
            
            # Form for new entry
            st.subheader("🏠 Resident Details Form")
            with st.form("entry_form"):
                building_no = st.text_input("Building No")
                door_no = st.text_input("Door No")
                name = st.text_input("Resident Name")
                contact_no = st.text_input("Contact Number")
                submit = st.form_submit_button("Save Entry")

                if submit:
                    with open(CSV_FILE, mode="a", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=FIELDS)
                        writer.writerow({
                            "building_no": building_no,
                            "door_no": door_no,
                            "name": name,
                            "contact_no": contact_no,
                            "car_color": color_name,
                            "car_numberplate": number_plate,
                            "car_status": car_status
                        })

                    st.success("✅ Entry saved successfully!")
                    st.write("### 🔍 Entry Summary")
                    st.write({
                        "Building No": building_no,
                        "Door No": door_no,
                        "Name": name,
                        "Contact No": contact_no,
                        "Car Color": color_name,
                        "Number Plate": number_plate,
                        "Status": car_status
                    })

    with st.expander("📋 View All Saved Entries"):
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            st.dataframe(df)
        else:
            st.info("No entries saved yet.")

if __name__ == "__main__":
    main()
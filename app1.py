
import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import cv2
import os
import requests
from dotenv import load_dotenv
from graph_visualization import plot_macronutrient_pie  # Import the new function

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
USDA_API_KEY = os.getenv("USDA_API_KEY")

# Hugging Face API Models
TEXT_GEN_MODEL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
IMAGE_CAPTION_MODEL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"

# Set Tesseract OCR path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Function to fetch food nutrition data from USDA API
def get_food_nutrition(food_name):
    url = f"{USDA_API_URL}?query={food_name}&api_key={USDA_API_KEY}"
    response = requests.get(url)
    st.write(f"Fetching nutrition data for: {food_name}")
    data = response.json()
    
    if "foods" in data and len(data["foods"]) > 0:
        nutrients = data["foods"][0]["foodNutrients"]
        
        calories = next((n["value"] for n in nutrients if n["nutrientName"] == "Energy"), 0)
        protein = next((n["value"] for n in nutrients if "Protein" in n["nutrientName"]), 0)
        carbs = next((n["value"] for n in nutrients if "Carbohydrate" in n["nutrientName"]), 0)
        fats = next((n["value"] for n in nutrients if "Total lipid" in n["nutrientName"]), 0)
        
        vitamins = [n["nutrientName"] for n in nutrients if "Vitamin" in n["nutrientName"]]

        return {
            "Fetched Calories": calories,
            "Protein": protein,
            "Carbs": carbs,
            "Fats": fats,
            "Vitamins": ", ".join(vitamins),
        }
    
    return {"Fetched Calories": 0, "Protein": 0, "Carbs": 0, "Fats": 0, "Vitamins": ""}

# Function to extract text from a table image
def extract_table_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    extracted_text = pytesseract.image_to_string(thresh)
    
    lines = extracted_text.split("\n")
    data = [line.split() for line in lines if line.strip()]
    
    if len(data) > 1:
        df = pd.DataFrame(data[1:], columns=["Meal", "Food", "Calories"])
        df.to_csv("extracted_diet.csv", index=False)
        return df
    return None

# Hugging Face AI Meal Analysis
def get_food_analysis(meal, food, calories):
    prompt = f"Analyze {meal}: {food}, {calories} calories. Give health benefits in points format with emojis."
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    data = {"inputs": prompt}
    
    st.write(f"Generating analysis for: {meal} - {food}")
    response = requests.post(TEXT_GEN_MODEL, headers=headers, json=data)
    result = response.json()
    
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "No response available.")
    return "Unexpected API response format."

# Function to calculate calories burned
def calculate_calories_burned(steps, weight, height):
    MET = 3.5  # Approximate MET value for walking
    calories_burned = (MET * weight * steps * 0.0005)
    return int(calories_burned)

# Streamlit Sidebar
st.sidebar.title("🍽️ Diet & Fitness Settings")
daily_calorie_goal = st.sidebar.number_input("Set daily calorie goal (kcal):", min_value=500, max_value=5000, value=2000, step=100)
steps = st.sidebar.number_input("Steps Walked:", min_value=0, value=5000, step=100)
weight = st.sidebar.number_input("Weight (kg):", min_value=30.0, value=70.0, step=0.1)
height = st.sidebar.number_input("Height (cm):", min_value=100.0, value=170.0, step=1.0)

diet_file = st.sidebar.file_uploader("Upload a Diet Plan (Image, Food Photo, or CSV)", type=["csv", "jpg", "jpeg", "png"])

burned_calories = calculate_calories_burned(steps, weight, height)
st.sidebar.write(f"🔥 Estimated Calories Burned: **{burned_calories} kcal**")

adjusted_calorie_goal = daily_calorie_goal - burned_calories
st.sidebar.write(f"📌 Adjusted Daily Calorie Intake: **{adjusted_calorie_goal} kcal**")

if adjusted_calorie_goal < 1500:
    st.sidebar.warning("⚠️ Your calorie intake is too low. Consider increasing your food intake.")
elif adjusted_calorie_goal > 3000:
    st.sidebar.warning("⚠️ Your calorie intake is high. Consider reducing calorie-dense foods.")
else:
    st.sidebar.success("✅ Your diet is well-balanced!")

df = None
if diet_file is not None:
    if diet_file.type in ["image/jpeg", "image/png"]:
        image = Image.open(diet_file)
        image_path = "uploaded_image.jpg"
        image.save(image_path)
        df = extract_table_from_image(image_path)

        if df is not None:
            st.markdown("## 📋 Personalized Diet Plan")
            st.write(f"**Daily Calorie Goal:** {daily_calorie_goal} kcal")

            nutrition_data = []
            for _, row in df.iterrows():
                food_nutrition = get_food_nutrition(row["Food"])
                nutrition_data.append(food_nutrition)

            nutrition_df = pd.DataFrame(nutrition_data)

            # Rename 'Calories' to 'Fetched Calories' before merging
            nutrition_df.rename(columns={"Calories": "Fetched Calories"}, inplace=True)

            # Merge and drop duplicate columns
            df = pd.concat([df.drop(columns=["Calories"]), nutrition_df], axis=1)

            total_default_calories = df["Fetched Calories"].sum()
            calorie_ratio = adjusted_calorie_goal / total_default_calories if total_default_calories > 0 else 1

            df["Adjusted Calories"] = (df["Fetched Calories"] * calorie_ratio).astype(int)

            # Remove the Minerals column if it exists and adjust the Vitamins column size
            if "Minerals" in df.columns:
                df.drop(columns=["Minerals"], inplace=True)

            # Customize the display settings of the DataFrame
            st.markdown(
                """
                <style>
                .dataframe-table {
                    font-size: 12px;  /* Reduce the font size */
                }
                .dataframe-table th {
                    font-size: 14px;  /* Reduce the font size for headers */
                }
                .dataframe-table td {
                    font-size: 12px;  /* Reduce the font size for cells */
                }
                .dataframe-table td:nth-child(6) {
                    max-width: 150px;  /* Set the max width for Vitamins column */
                }
                .dataframe-table th:nth-child(6) {
                    max-width: 150px;  /* Set the max width for Vitamins column */
                }
                .dataframe-table th:nth-child(n+7) {
                    display: none;  /* Hide the columns after the sixth one */
                }
                .dataframe-table td:nth-child(n+7) {
                    display: none;  /* Hide the cells after the sixth one */
                }
                </style>
                """, unsafe_allow_html=True)

            # Display the updated DataFrame
            st.markdown("## 🍽️ Diet Plan with Macronutrients")
            st.write(df[["Meal", "Food", "Adjusted Calories", "Protein", "Carbs", "Fats", "Vitamins"]])

            st.markdown("## 🤖 AI Nutrition Analysis")
            for _, row in df.iterrows():
                st.markdown(f"### **{row['Meal']}**")
                analysis = get_food_analysis(row["Meal"], row["Food"], row["Adjusted Calories"])
                st.write(analysis)

            # Plot the macronutrient pie chart
            plot_macronutrient_pie(df)

        else:
            st.markdown("## 📸 Food Image Uploaded")
            st.image(image, caption="Uploaded Dish", use_column_width=True)
    else:
        df = pd.read_csv(diet_file)
        st.write("CSV file loaded successfully.")
       
        df["Calories"] = df["Calories"].astype(float)
        
        # Fetch and add micronutrient data for each food item
        nutrition_data = []
        for _, row in df.iterrows():
            food_nutrition = get_food_nutrition(row["Food"])
            nutrition_data.append(food_nutrition)

        nutrition_df = pd.DataFrame(nutrition_data)
        
        # Merge and drop duplicate columns
        df = pd.concat([df.drop(columns=["Calories"]), nutrition_df], axis=1)

        total_default_calories = df["Fetched Calories"].sum()
        calorie_ratio = adjusted_calorie_goal / total_default_calories if total_default_calories > 0 else 1
        df["Adjusted Calories"] = (df["Fetched Calories"] * calorie_ratio).astype(int)

        # Remove the Minerals column if it exists and adjust the Vitamins column size
        if "Minerals" in df.columns:
            df.drop(columns=["Minerals"], inplace=True)

        # Customize the display settings of the DataFrame
        st.markdown(
            """
            <style>
            .dataframe-table {
                font-size: 12px;  /* Reduce the font size */
            }
            .dataframe-table th {
                font-size: 14px;  /* Reduce the font size for headers */
            }
            .dataframe-table td {
                font-size: 12px;  /* Reduce the font size for cells */
            }
            .dataframe-table td:nth-child(6) {
                max-width: 150px;  /* Set the max width for Vitamins column */
            }
            .dataframe-table th:nth-child(6) {
                max-width: 150px;  /* Set the max width for Vitamins column */
            }
            .dataframe-table th:nth-child(n+7) {
                display: none;  /* Hide the columns after the sixth one */
            }
            .dataframe-table td:nth-child(n+7) {
                display: none;  /* Hide the cells after the sixth one */
            }
            </style>
            """, unsafe_allow_html=True)

        # Display the updated DataFrame
        st.markdown("## 🍽️ Diet Plan with Macronutrients")
        st.write(df[["Meal", "Food", "Adjusted Calories", "Protein", "Carbs", "Fats", "Vitamins"]])

        st.markdown("## 🤖 AI Nutrition Analysis")
        for _, row in df.iterrows():
            st.markdown(f"### **{row['Meal']}**")
            analysis = get_food_analysis(row["Meal"], row["Food"], row["Adjusted Calories"])
            st.write(analysis)

        # Plot the macronutrient pie chart
        plot_macronutrient_pie(df)
else:
    st.info("Please upload a diet plan image, a food photo, or a CSV file.")

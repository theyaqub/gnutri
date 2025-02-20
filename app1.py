import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import cv2
import numpy as np
import google.generativeai as genai
import os
import time
import random
import textwrap
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

# Load environment variables
load_dotenv()

# Retrieve API key
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Missing API Key! Check your .env file.")
    st.stop()

# Configure Gemini AI
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Set Tesseract OCR path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Cache for storing API responses
cache = {}

# Function to get AI-based food analysis (Optimized)
def get_gemini_food_analysis(meal, food, calories):
    prompt = f"Analyze this meal: {meal}, containing {food}. It has {calories} calories. Give a brief health analysis (max 600 words)."

    # Return cached result if available
    if (meal, food, calories) in cache:
        return cache[(meal, food, calories)]

    # Rate limit: Wait 2-5 seconds between API calls
    time.sleep(random.uniform(2, 5))

    try:
        response = model.generate_content([prompt])
        if response and response.text:
            analysis = textwrap.shorten(response.text, width=4500, placeholder="...")
            cache[(meal, food, calories)] = analysis  # Cache result
            return analysis
        return "No response from AI."

    except ResourceExhausted:
        return "⚠️ API quota exceeded. Please try again later."

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Sidebar
st.sidebar.title("Diet Plan Settings")
daily_calorie_goal = st.sidebar.number_input("Set your daily calorie goal (kcal):", min_value=500, max_value=5000, value=2000, step=100)

# File uploader
diet_file = st.sidebar.file_uploader("Upload your diet plan (Image, Food Photo, or CSV)", type=["csv", "jpg", "jpeg", "png"])

df = None
if diet_file is not None:
    if diet_file.type in ["image/jpeg", "image/png"]:
        image = Image.open(diet_file)
        image_path = "uploaded_image.jpg"
        image.save(image_path)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.subheader("🍽️ AI-Generated Food Description")

        food_description = get_gemini_food_analysis("Unknown Meal", "Unknown Food", "Unknown Calories")
        st.write(food_description)

    else:
        df = pd.read_csv(diet_file)

        if "Meal" in df.columns and "Food" in df.columns and "Calories" in df.columns:
            df["Calories"] = df["Calories"].astype(float)
            total_default_calories = df["Calories"].sum()
            calorie_ratio = daily_calorie_goal / total_default_calories
            df["Adjusted Calories"] = (df["Calories"] * calorie_ratio).astype(int)

            st.header("🍎 Personalized Diet Plan")
            st.write(f"**Daily Calorie Goal:** {daily_calorie_goal} kcal")
            st.table(df[["Meal", "Food", "Adjusted Calories"]])

            # AI-based meal analysis (batching requests)
            st.subheader("🤖 AI Nutrition Analysis")

            # Group meals to avoid duplicate calls
            unique_meals = df.drop_duplicates(subset=["Meal", "Food", "Adjusted Calories"])

            for _, row in unique_meals.iterrows():
                analysis = get_gemini_food_analysis(row["Meal"], row["Food"], row["Adjusted Calories"])
                st.write(f"**{row['Meal']}:** {analysis}")

        else:
            st.error("CSV must contain 'Meal', 'Food', and 'Calories' columns.")

else:
    st.info("Please upload a diet plan image, a food photo, or a CSV file.")

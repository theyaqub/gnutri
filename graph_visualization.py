import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Function to plot pie charts for macronutrients
def plot_macronutrient_pie(df):
    try:
        # Aggregating macronutrients
        nutrient_totals = df[['Protein', 'Carbs', 'Fats']].sum()

        # Plotting Pie Chart
        fig, ax = plt.subplots()
        ax.pie(nutrient_totals, labels=nutrient_totals.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Macronutrient Distribution")
        
        # Display the pie chart using Streamlit
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating pie chart: {e}")

# Function to generate data frame for testing (optional)
def create_sample_df():
    data = {
        "Meal": ["Breakfast", "Lunch", "Dinner"],
        "Food": ["Eggs", "Chicken Breast", "Salmon"],
        "Protein": [12, 35, 30],
        "Carbs": [1, 0, 0],
        "Fats": [11, 3.6, 13]
    }
    return pd.DataFrame(data)

# Test the function (uncomment this block if you want to run it standalone)
# if __name__ == "__main__":
#     df = create_sample_df()
#     plot_macronutrient_pie(df)

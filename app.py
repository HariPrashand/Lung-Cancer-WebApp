import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("BMI Calculator")

# Instructions
st.write("Calculate your Body Mass Index (BMI) based on your weight and height.")

# Input for weight
weight = st.number_input("Enter your weight (kg):", min_value=1.0, max_value=200.0, step=0.1)

# Input for height
height = st.number_input("Enter your height (cm):", min_value=50.0, max_value=250.0, step=0.1)

# Calculate BMI
if height and weight:
    bmi = weight / ((height / 100) ** 2)
    st.write(f"Your BMI is: **{bmi:.2f}**")

    # BMI classification
    if bmi < 18.5:
        st.write("You are underweight.")
    elif 18.5 <= bmi < 24.9:
        st.write("You have a normal weight.")
    elif 25 <= bmi < 29.9:
        st.write("You are overweight.")
    else:
        st.write("You are obese.")
    
    # Plot BMI category
    categories = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
    values = [18.5, 24.9, 29.9, 40]

    fig, ax = plt.subplots()
    ax.barh(categories, values, color=['lightblue', 'lightgreen', 'orange', 'red'])
    ax.axvline(bmi, color='blue', label=f'Your BMI: {bmi:.2f}')
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Please enter valid weight and height.")

# Show a motivational quote
st.subheader("Stay healthy and take care of your body!")

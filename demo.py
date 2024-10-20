import streamlit as st
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


lung_cancer_df = pd.read_csv("lung_cancer_exported.csv")

temp_df = lung_cancer_df[['AGE', 'SMOKING', 'YELLOW_FINGERS', 'FATIGUE ', 'ALLERGY ', 'COUGHING',
              'CHRONIC DISEASE', 'SWALLOWING DIFFICULTY', 'CHEST PAIN',
              'LUNG_CANCER']]


# Apply label encoding to binary categorical features
label_encoder = LabelEncoder()

# Encode 'GENDER' and 'LUNG_CANCER' columns
for i in ['AGE','SMOKING', 'YELLOW_FINGERS', 'FATIGUE ', 'ALLERGY ', 'COUGHING',
              'CHRONIC DISEASE', 'SWALLOWING DIFFICULTY', 'CHEST PAIN',
              'LUNG_CANCER']:
  temp_df[i] = label_encoder.fit_transform(temp_df[i])


# Define feature columns (excluding the target column 'LUNG_CANCER')
X = temp_df.drop(columns=['LUNG_CANCER'])
y = temp_df['LUNG_CANCER']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]

# Define severity based on probability
def get_severity(prob):
    if prob < 0.4:
        return "Low Risk"
    elif prob < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

# Title
st.title("Lung Cancer Prediction")

# Inputs
age = st.number_input("Enter Age:", min_value=0, max_value=120, step=1)
smoking = st.radio("Do you smoke?", ['Yes', 'No'])
yellow = st.radio("Do you have yellow fingers?", ['Yes', 'No'])
fatigue = st.radio("Do you feel tired all the time?", ['Yes', 'No'])
allergy = st.radio("Are you allergic to anything?", ['Yes', 'No'])
cough = st.radio("Is your coughing more than your usual self?", ['Yes', 'No'])
chronic = st.radio("Are you currently or likely to be acquainted with any chronic disease?", ['Yes', 'No'])
swallowing = st.radio("Do you have difficulty in swallowing?", ['Yes', 'No'])
chest = st.radio("Do you have frequent chest pain?", ['Yes', 'No'])

# Convert 'Yes' to 1 and 'No' to 0
smoking = 1 if smoking == 'Yes' else 0
yellow = 1 if yellow == 'Yes' else 0
fatigue = 1 if fatigue == 'Yes' else 0
allergy = 1 if allergy == 'Yes' else 0
cough = 1 if cough == 'Yes' else 0
chronic = 1 if chronic == 'Yes' else 0
swallowing = 1 if swallowing == 'Yes' else 0
chest = 1 if chest == 'Yes' else 0

# After getting inputs, convert to format model can understand
# For example, map gender to binary value

# Predict and show severity based on inputs
if st.button("Predict"):
    # Prepare inputs as DataFrame and make prediction
    input_data = pd.DataFrame([[age, smoking, yellow, fatigue, allergy, cough, chronic , swallowing, chest]], columns=X_train.columns)
    pred = clf.predict(input_data)
    prob = clf.predict_proba(input_data)[:, 1]
    
    st.write(f"prob: {prob}")
    severity = get_severity(prob)
    
    # Display result
    st.write(f"Prediction: {'Lung Cancer' if pred == 1 else 'No Lung Cancer'}")
    st.write(f"Risk Level: {severity}")

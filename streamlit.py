import joblib
import numpy as np
import pandas as pd
import streamlit as st

file = open("RandomForestRegressor.joblib", "rb")
model = joblib.load(file)


# Function to get prediction and confidence score
def get_prediction_and_confidence(prediction_model, data):
    # Predict the class
    predicted_class = prediction_model.predict(data)
    # Get confidence scores (probabilities)
    # confidence_scores = prediction_model.predict_proba(data)
    # Get the confidence score for the predicted class
    # conf_score = confidence_scores[0][predicted_class[0]]
    return predicted_class[0]


def prep_input(arr):
    arr = pd.DataFrame(arr, columns=["value_eur", "movement_reaction", "age", "wage_eur",
                                     "potential", "international_reputation", "skill_moves", "defending",
                                     "skill_ball_control", "dribbling"])

    # convert columns to appropriate data types for the data frame
    arr["movement_reaction"] = arr["movement_reaction"].astype("int64")
    arr["age"] = arr["age"].astype("int64")
    arr["potential"] = arr["potential"].astype("int64")
    arr["international_reputation"] = arr["international_reputation"].astype("int64")
    arr["skill_moves"] = arr["skill_moves"].astype("int64")
    arr["skill_ball_control"] = arr["skill_ball_control"].astype("int64")

    return arr


# Title and Description
st.title("Player Rating Prediction")
st.write("""
This app predicts the player's rating based on their profile information.
""")

# Player Profile Input Section
st.header("Enter Player Profile")
player_name = st.text_input("Player Name")

age = st.number_input("Age", min_value=15, max_value=50, step=1)
potential = st.number_input(
    "Potential (Out of 100)", min_value=0, max_value=100, step=1)
value_eur = st.number_input("Value (EUR)")
international_reputation = st.number_input(
    "International Reputation(0-5)", min_value=0, max_value=5, step=1)
wage_eur = st.number_input("Wage (EUR)", min_value=0.0, value=0.0, step=0.01)
movement_reaction = st.number_input(
    "Movement Reaction (out of 100)", min_value=0, max_value=100, step=1)
skill_moves = st.number_input(
    "Skill Moves (0-5)", min_value=0, max_value=5, step=1)
defending = st.number_input("Defending", min_value=0, max_value=100, step=1)
skill_ball_control = st.number_input(
    "Skill Ball Moves (out of 100)", min_value=0, max_value=100, step=1)
dribbling = st.number_input(
    "Dribbling (out of 100)", min_value=0, max_value=100, step=1)

# Prepare input data for prediction
input_data = np.array([[value_eur, movement_reaction, age, wage_eur, potential, international_reputation, skill_moves,
                        defending, skill_ball_control, dribbling]])

# Predict Button
if st.button("Predict Rating"):
    input_data = prep_input(input_data)

    # Get prediction and confidence score
    predicted_rating = get_prediction_and_confidence(model, input_data)

    # Prediction Output Section
    st.header("Prediction Result")
    st.subheader("Welcome, " + player_name)
    st.markdown("-----------------------------------------------")
    st.write(f"**Predicted Rating:** {predicted_rating}")
    # st.write(f"**Confidence Level:** {confidence_score * 100:.2f}%")


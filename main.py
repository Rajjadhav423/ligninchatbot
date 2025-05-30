
# import numpy as np
# import pandas as pd
# import pickle
# # import pickle
# # import numpy as np

# # Step 1: Load the model from file
# with open("all (1).pkl", "rb") as f:
#     model = pickle.load(f)

# # Step 2: Prepare test input
# # Suppose temperature = 25°C, salinity = 35 PSU
# X_test = np.array([[25, 35]])

# # Step 3: Predict using the model
# output = model.predict(X_test)[0]  # Get the first prediction row

# labels = [
#     ("Seawater Density (kg/m³)", "kg/m³"),
#     ("Ultrasonic Velocity (m/s)", "m/s"),
#     ("Thermal Expansion Coefficient (K⁻¹)", "K⁻¹"),
#     ("Adiabatic Compressibility (TPa⁻¹)", "TPa⁻¹"),
#     ("Isothermal Compressibility (TPa⁻¹)", "TPa⁻¹"),
#     ("Heat Capacity (kJ/kg·K)", "kJ/kg·K"),
#     ("Intermolecular Free Length (×10⁻¹¹ m)", "×10⁻¹¹ m"),
#     ("Internal Pressure (MPa)", "MPa"),
#     ("Cohesion Energy Density (Pa·m)", "Pa·m"),
#     ("Grüneisen Parameter", ""),
#     ("Acoustic Impedance (×10⁴ kg/m²·s)", "×10⁴ kg/m²·s"),
#     ("Non-Linearity Parameter", "")
# ]


# # Step 5: Print all outputs with labels
# print("🔍 Predicted Seawater Properties:\n")
# for i, (label, unit) in enumerate(labels):
#     print(f"{label}: {output[i]:.6f} {unit}")





# # Load the model
# with open("ice_rf_model.pkl", "rb") as f:
#     model = pickle.load(f)

# # Print expected feature names (optional, for debug)
# #input are temperature , year , months
# print("Model expects features:", model.feature_names_in_)

# # Prepare test input as DataFrame with correct feature names
# X_test = pd.DataFrame([[25, 35, 10]], columns=model.feature_names_in_)

# # Predict
# output = model.predict(X_test)``


# print("Prediction output:ice melting point", output)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()
# Add CORS middleware — allow everything (for development only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Load models once at startup
with open("all (1).pkl", "rb") as f:
    seawater_model = pickle.load(f)

with open("ice_rf_model.pkl", "rb") as f:
    ice_model = pickle.load(f)


# Input schema for seawater model
class SeawaterInput(BaseModel):
    temperature: float
    salinity: float

# Input schema for ice melting point model
class IceInput(BaseModel):
    temperature: float
    year: int
    month: int


@app.post("/predict_seawater")
def predict_seawater(data: SeawaterInput):
    # Prepare input as numpy array
    X_test = np.array([[data.temperature, data.salinity]])
    output = seawater_model.predict(X_test)[0]
    
    # Labels as per your original code
    labels = [
        "Seawater Density (kg/m³)",
        "Ultrasonic Velocity (m/s)",
        "Thermal Expansion Coefficient (K⁻¹)",
        "Adiabatic Compressibility (TPa⁻¹)",
        "Isothermal Compressibility (TPa⁻¹)",
        "Heat Capacity (kJ/kg·K)",
        "Intermolecular Free Length (×10⁻¹¹ m)",
        "Internal Pressure (MPa)",
        "Cohesion Energy Density (Pa·m)",
        "Grüneisen Parameter",
        "Acoustic Impedance (×10⁴ kg/m²·s)",
        "Non-Linearity Parameter"
    ]

    # Return dict with labels and predictions
    result = {label: float(value) for label, value in zip(labels, output)}
    return {"prediction": result}


@app.post("/predict_ice")
def predict_ice(data: IceInput):
    # Prepare input as numpy array with correct shape
    # Assuming ice_model expects features in order: temperature, year, month
    X_test = np.array([[data.temperature, data.year, data.month]])
    output = ice_model.predict(X_test)[0]
    return {"ice_melting_point": float(output)}

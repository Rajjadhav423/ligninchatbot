
# # import numpy as np
# # import pandas as pd
# # import pickle
# # # import pickle
# # # import numpy as np

# # # Step 1: Load the model from file
# # with open("all (1).pkl", "rb") as f:
# #     model = pickle.load(f)

# # # Step 2: Prepare test input
# # # Suppose temperature = 25°C, salinity = 35 PSU
# # X_test = np.array([[25, 35]])

# # # Step 3: Predict using the model
# # output = model.predict(X_test)[0]  # Get the first prediction row

# # labels = [
# #     ("Seawater Density (kg/m³)", "kg/m³"),
# #     ("Ultrasonic Velocity (m/s)", "m/s"),
# #     ("Thermal Expansion Coefficient (K⁻¹)", "K⁻¹"),
# #     ("Adiabatic Compressibility (TPa⁻¹)", "TPa⁻¹"),
# #     ("Isothermal Compressibility (TPa⁻¹)", "TPa⁻¹"),
# #     ("Heat Capacity (kJ/kg·K)", "kJ/kg·K"),
# #     ("Intermolecular Free Length (×10⁻¹¹ m)", "×10⁻¹¹ m"),
# #     ("Internal Pressure (MPa)", "MPa"),
# #     ("Cohesion Energy Density (Pa·m)", "Pa·m"),
# #     ("Grüneisen Parameter", ""),
# #     ("Acoustic Impedance (×10⁴ kg/m²·s)", "×10⁴ kg/m²·s"),
# #     ("Non-Linearity Parameter", "")
# # ]


# # # Step 5: Print all outputs with labels
# # print("🔍 Predicted Seawater Properties:\n")
# # for i, (label, unit) in enumerate(labels):
# #     print(f"{label}: {output[i]:.6f} {unit}")





# # # Load the model
# # with open("ice_rf_model.pkl", "rb") as f:
# #     model = pickle.load(f)

# # # Print expected feature names (optional, for debug)
# # #input are temperature , year , months
# # print("Model expects features:", model.feature_names_in_)

# # # Prepare test input as DataFrame with correct feature names
# # X_test = pd.DataFrame([[25, 35, 10]], columns=model.feature_names_in_)

# # # Predict
# # output = model.predict(X_test)``


# # print("Prediction output:ice melting point", output)
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import pickle

# app = FastAPI()
# # Add CORS middleware — allow everything (for development only)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )
# # Load models once at startup
# with open("all (1).pkl", "rb") as f:
#     seawater_model = pickle.load(f)

# with open("ice_rf_model.pkl", "rb") as f:
#     ice_model = pickle.load(f)


# # Input schema for seawater model
# class SeawaterInput(BaseModel):
#     temperature: float
#     salinity: float

# # Input schema for ice melting point model
# class IceInput(BaseModel):
#     temperature: float
#     year: int
#     month: int


# @app.post("/predict_seawater")
# def predict_seawater(data: SeawaterInput):
#     # Prepare input as numpy array
#     X_test = np.array([[data.temperature, data.salinity]])
#     output = seawater_model.predict(X_test)[0]
    
#     # Labels as per your original code
#     labels = [
#         "Seawater Density (kg/m³)",
#         "Ultrasonic Velocity (m/s)",
#         "Thermal Expansion Coefficient (K⁻¹)",
#         "Adiabatic Compressibility (TPa⁻¹)",
#         "Isothermal Compressibility (TPa⁻¹)",
#         "Heat Capacity (kJ/kg·K)",
#         "Intermolecular Free Length (×10⁻¹¹ m)",
#         "Internal Pressure (MPa)",
#         "Cohesion Energy Density (Pa·m)",
#         "Grüneisen Parameter",
#         "Acoustic Impedance (×10⁴ kg/m²·s)",
#         "Non-Linearity Parameter"
#     ]

#     # Return dict with labels and predictions
#     result = {label: float(value) for label, value in zip(labels, output)}
#     return {"prediction": result}


# @app.post("/predict_ice")
# def predict_ice(data: IceInput):
#     # Prepare input as numpy array with correct shape
#     # Assuming ice_model expects features in order: temperature, year, month
#     X_test = np.array([[data.temperature, data.year, data.month]])
#     output = ice_model.predict(X_test)[0]
#     return {"ice_melting_point": float(output)}

# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Seawater Properties Prediction API",
    description="API for predicting seawater properties based on temperature and salinity",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model
try:
    with open('seawater_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    print("Model loaded successfully!")
    print(f"Model R² Score: {model_data['test_r2_score']:.4f}")
except FileNotFoundError:
    print("Error: seawater_model.pkl not found. Please run the model training script first.")
    model_data = None

# Request model for input validation
class SeawaterInput(BaseModel):
    temperature_k: float = Field(
        ..., 
        ge=273.15, 
        le=373.15, 
        description="Temperature in Kelvin (273.15 - 373.15 K)"
    )
    salinity_g_kg: float = Field(
        ..., 
        ge=0, 
        le=50, 
        description="Salinity in g/kg (0 - 50 g/kg)"
    )

    class Config:
        schema_extra = {
            "example": {
                "temperature_k": 297.15,
                "salinity_g_kg": 20.0
            }
        }

# Response model
class SeawaterOutput(BaseModel):
    input_parameters: Dict[str, float]
    predictions: Dict[str, float]
    model_info: Dict[str, Any]

def predict_seawater_properties(temperature_k: float, salinity_g_kg: float) -> Dict[str, float]:
    """
    Predict seawater properties given temperature and salinity
    """
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Create input with enhanced features
    input_data = pd.DataFrame({
        'Temperature (K)': [temperature_k],
        'Salinity (g/kg)': [salinity_g_kg]
    })
    
    # Add engineered features
    input_enhanced = input_data.copy()
    input_enhanced['Temp_Sal_Interaction'] = input_data['Temperature (K)'] * input_data['Salinity (g/kg)']
    input_enhanced['Temp_Squared'] = input_data['Temperature (K)'] ** 2
    input_enhanced['Sal_Squared'] = input_data['Salinity (g/kg)'] ** 2
    input_enhanced['Temp_Cubed'] = input_data['Temperature (K)'] ** 3
    input_enhanced['Sal_Cubed'] = input_data['Salinity (g/kg)'] ** 3
    input_enhanced['Temp_Sal_Squared'] = input_data['Temperature (K)'] * (input_data['Salinity (g/kg)'] ** 2)
    input_enhanced['Temp_Squared_Sal'] = (input_data['Temperature (K)'] ** 2) * input_data['Salinity (g/kg)']
    
    # Scale features
    input_scaled = model_data['scaler'].transform(input_enhanced)
    
    # Predict
    prediction = model_data['model'].predict(input_scaled)
    
    # Create result dictionary
    result = {}
    for i, target_name in enumerate(model_data['target_names']):
        result[target_name] = float(prediction[0][i])
    
    return result

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Seawater Properties Prediction API",
        "status": "active",
        "model_loaded": model_data is not None,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model-info": "/model-info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "model_r2_score": model_data['test_r2_score'] if model_data else None
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "Multi-Output Gradient Boosting Regressor",
        "feature_count": len(model_data['feature_names']),
        "target_count": len(model_data['target_names']),
        "feature_names": model_data['feature_names'],
        "target_names": model_data['target_names'],
        "test_r2_score": model_data['test_r2_score'],
        "input_requirements": {
            "temperature_k": "Temperature in Kelvin (273.15 - 373.15)",
            "salinity_g_kg": "Salinity in g/kg (0 - 50)"
        }
    }

@app.post("/predict", response_model=SeawaterOutput)
async def predict_properties(input_data: SeawaterInput):
    """
    Predict seawater properties based on temperature and salinity
    
    Returns predictions for all seawater properties including:
    - Density (kg/m3)
    - Ultrasonic Velocity (m/s)
    - Acoustic Impedance (Z) (kg/m2s) x104
    - Internal Pressure (πi) (MPa)
    - Heat Capacity (Cp)(kj/kgK)
    - Cohesion Energy Density (CED) (MPa)
    - And other properties...
    """
    try:
        # Make prediction
        predictions = predict_seawater_properties(
            input_data.temperature_k, 
            input_data.salinity_g_kg
        )
        
        # Prepare response
        response = SeawaterOutput(
            input_parameters={
                "temperature_k": input_data.temperature_k,
                "salinity_g_kg": input_data.salinity_g_kg,
                "temperature_celsius": input_data.temperature_k - 273.15
            },
            predictions=predictions,
            model_info={
                "model_type": "Multi-Output Gradient Boosting Regressor",
                "r2_score": model_data['test_r2_score'],
                "prediction_count": len(predictions)
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(input_list: list[SeawaterInput]):
    """
    Predict seawater properties for multiple input combinations
    """
    if len(input_list) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")
    
    results = []
    for input_data in input_list:
        try:
            predictions = predict_seawater_properties(
                input_data.temperature_k, 
                input_data.salinity_g_kg
            )
            
            results.append({
                "input": {
                    "temperature_k": input_data.temperature_k,
                    "salinity_g_kg": input_data.salinity_g_kg
                },
                "predictions": predictions
            })
        except Exception as e:
            results.append({
                "input": {
                    "temperature_k": input_data.temperature_k,
                    "salinity_g_kg": input_data.salinity_g_kg
                },
                "error": str(e)
            })
    
    return {
        "batch_size": len(input_list),
        "results": results,
        "model_info": {
            "r2_score": model_data['test_r2_score'] if model_data else None
        }
    }

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main:app",  # Change this to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True
    )
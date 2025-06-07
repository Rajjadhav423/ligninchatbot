
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

# # with open("ice_melting_api_model.pkl", "rb") as f:
# #     ice_model = pickle.load(f)
# with open("ice_melting_api_model.pkl", "rb") as f:
#     ice_model_dict = pickle.load(f)
#     ice_model = ice_model_dict["model"]

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

# demo.py
from fastapi import FastAPI, HTTPException ,Query
from pydantic import BaseModel, Field
import numpy as np
import pickle
import joblib
import os
from typing import Dict, Any
import uvicorn
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Ice Melting Rate Prediction API",
    description="API for predicting ice melting rates based on temperature, year, and month",
    version="1.0.0"
)

# Global variable to store the loaded model
model_data = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    temperature_k: float = Field(..., description="Temperature in Kelvin", ge=200.0, le=300.0)
    year: int = Field(..., description="Year", ge=1990, le=2030)
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)

# class PredictionResponse(BaseModel):
#     predicted_ice_melting_rate: float = Field(..., description="Predicted ice melting rate in Gton/month")
#     temperature_k: float
#     year: int
#     month: int
#     model_used: str
#     prediction_timestamp: str
class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # add this line

    predicted_ice_melting_rate: float
    temperature_k: float
    year: int
    month: int
    model_used: str
    prediction_timestamp: str
class ModelInfo(BaseModel):
    model_name: str
    test_r2_score: float
    cv_score: float
    rmse: float
    mae: float
    features_used: list
    data_range: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool

def load_model():
    """Load the trained model from file"""
    global model_data
    
    # Try to load from different possible locations
    model_paths = [
        "ice_melting_api_model.pkl",
        "ice_melting_api_model.joblib"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                if model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                else:
                    model_data = joblib.load(model_path)
                print(f"✅ Model loaded successfully from: {model_path}")
                return True
            except Exception as e:
                print(f"❌ Error loading model from {model_path}: {str(e)}")
                continue
    
    print("❌ No model file found in any of the expected locations")
    return False

def create_features(temperature_k: float, year: int, month: int) -> np.ndarray:
    """
    Create feature array with the same engineering as training
    """
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Get normalization factors from model data
    year_min = model_data['feature_engineering_info']['normalization_factors']['year_min']
    year_max = model_data['feature_engineering_info']['normalization_factors']['year_max']
    
    # Feature engineering (same as training)
    year_norm = (year - year_min) / (year_max - year_min)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    temp_squared = temperature_k ** 2
    temp_year_interaction = temperature_k * year_norm
    temp_month_sin = temperature_k * month_sin
    temp_month_cos = temperature_k * month_cos
    
    # Create feature array in the same order as training
    features = np.array([[
        temperature_k, year, month, year_norm, month_sin, month_cos,
        temp_squared, temp_year_interaction, temp_month_sin, temp_month_cos
    ]])
    
    return features

def predict_ice_melting_rate(temperature_k: float, year: int, month: int) -> float:
    """
    Predict ice melting rate using the loaded model
    """
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create features
        features = create_features(temperature_k, year, month)
        
        # Apply scaling if the model requires it
        if model_data['uses_scaling'] and model_data['scaler'] is not None:
            features = model_data['scaler'].transform(features)
        
        # Make prediction
        prediction = model_data['model'].predict(features)[0]
        return float(prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        print("⚠️  Warning: Model not loaded. Some endpoints may not work.")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        message="Ice Melting Rate Prediction API is running",
        model_loaded=model_data is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_data is not None else "degraded",
        message="Model loaded successfully" if model_data is not None else "Model not loaded",
        model_loaded=model_data is not None
    )

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=model_data['model_name'],
        test_r2_score=model_data['performance']['test_r2_score'],
        cv_score=model_data['performance']['cv_score'],
        rmse=model_data['performance']['rmse'],
        mae=model_data['performance']['mae'],
        features_used=model_data['feature_columns'],
        data_range=model_data['data_stats']
    )

@app.post("/predict_ice", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict ice melting rate based on temperature, year, and month
    """
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        predicted_rate = predict_ice_melting_rate(
            request.temperature_k, 
            request.year, 
            request.month
        )
        
        return PredictionResponse(
            predicted_ice_melting_rate=round(predicted_rate, 2),
            temperature_k=request.temperature_k,
            year=request.year,
            month=request.month,
            model_used=model_data['model_name'],
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/predict")
async def predict_get(
    temperature_k: float = Query(..., description="Temperature in Kelvin", ge=200.0, le=300.0),
    year: int = Query(..., description="Year", ge=1990, le=2030),
    month: int = Query(..., description="Month (1-12)", ge=1, le=12)
):
    """
    GET endpoint for prediction (alternative to POST)
    """
    request = PredictionRequest(
        temperature_k=temperature_k,
        year=year,
        month=month
    )
    return await predict(request)

@app.post("/batch-predict")
async def batch_predict(requests: list[PredictionRequest]):
    """
    Batch prediction for multiple inputs
    """
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")
    
    results = []
    for req in requests:
        try:
            predicted_rate = predict_ice_melting_rate(req.temperature_k, req.year, req.month)
            results.append(PredictionResponse(
                predicted_ice_melting_rate=round(predicted_rate, 2),
                temperature_k=req.temperature_k,
                year=req.year,
                month=req.month,
                model_used=model_data['model_name'],
                prediction_timestamp=datetime.now().isoformat()
            ))
        except Exception as e:
            results.append({
                "error": f"Failed to predict for temp={req.temperature_k}, year={req.year}, month={req.month}: {str(e)}"
            })
    
    return {"predictions": results}

# Manual model loading endpoint (for debugging)
@app.post("/load-model")
async def load_model_endpoint():
    """
    Manually trigger model loading
    """
    success = load_model()
    if success:
        return {"message": "Model loaded successfully", "model_name": model_data['model_name']}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

if __name__ == "__main__":
    print("🚀 Starting Ice Melting Rate Prediction API...")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    
    # Run the server
    uvicorn.run(
        "demo:app",  # Change this to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
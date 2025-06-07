# main.py - Combined Seawater Properties and Ice Melting Rate Prediction API
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import uvicorn
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Combined Environmental Prediction API",
    description="API for predicting seawater properties and ice melting rates",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables for models
seawater_model_data = None
ice_model_data = None
multi_output_model_data = None

# ============= ADD NEW INPUT/OUTPUT MODELS =============
class MultiOutputInput(BaseModel):
    year: int = Field(..., description="Year", ge=1990, le=2030)
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)
    temperature_k: float = Field(..., description="Temperature in Kelvin", ge=200.0, le=350.0)

    class Config:
        schema_extra = {
            "example": {
                "year": 2024,
                "month": 6,
                "temperature_k": 275.5
            }
        }

class MultiOutputResponse(BaseModel):
    input_parameters: Dict[str, Any]
    predictions: Dict[str, float]
    model_info: Dict[str, Any]
    prediction_timestamp: str

# ============= ADD NEW MODEL LOADING FUNCTION =============
def load_multi_output_model():
    """Load the multi-output model"""
    global multi_output_model_data
    
    try:
        with open('multi_output_model.pkl', 'rb') as f:
            multi_output_model_data = pickle.load(f)
        print("✅ Multi-output model loaded successfully!")
        print(f"Model: {multi_output_model_data['model_name']}")
        print(f"Overall R² Score: {multi_output_model_data['performance']['overall_r2']:.4f}")
        return True
    except FileNotFoundError:
        print("❌ Error: multi_output_model.pkl not found.")
        return False
    except Exception as e:
        print(f"❌ Error loading multi-output model: {str(e)}")
        return False

# ============= ADD NEW PREDICTION FUNCTION =============
def predict_multi_output_properties(year: int, month: int, temperature_k: float) -> Dict[str, float]:
    """Predict all output properties using multi-output model"""
    if multi_output_model_data is None:
        raise HTTPException(status_code=500, detail="Multi-output model not loaded")
    
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'year': [year],
            'Month': [month],
            'Temperature (K)': [temperature_k]
        })
        
        # Fill any missing values with mean (same as training)
        input_data = input_data.fillna(input_data.mean())
        
        # Make prediction
        predictions = multi_output_model_data['model'].predict(input_data)
        
        # Create result dictionary
        result = {}
        for i, output_name in enumerate(multi_output_model_data['output_features']):
            result[output_name] = float(predictions[0][i])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-output prediction error: {str(e)}")



# ============= SEAWATER MODELS =============
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

class SeawaterOutput(BaseModel):
    input_parameters: Dict[str, float]
    predictions: Dict[str, float]
    model_info: Dict[str, Any]

# ============= ICE MELTING MODELS =============
class IcePredictionRequest(BaseModel):
    temperature_k: float = Field(..., description="Temperature in Kelvin", ge=200.0, le=300.0)
    year: int = Field(..., description="Year", ge=1990, le=2030)
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)

class IcePredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    predicted_ice_melting_rate: float
    temperature_k: float
    year: int
    month: int
    model_used: str
    prediction_timestamp: str

class ModelInfo(BaseModel):
    seawater_model: Optional[Dict[str, Any]] = None
    ice_model: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    seawater_model_loaded: bool
    ice_model_loaded: bool

# ============= MODEL LOADING FUNCTIONS =============
def load_seawater_model():
    """Load the seawater properties model"""
    global seawater_model_data
    
    try:
        with open('seawater_model.pkl', 'rb') as f:
            seawater_model_data = pickle.load(f)
        print("✅ Seawater model loaded successfully!")
        print(f"Seawater Model R² Score: {seawater_model_data['test_r2_score']:.4f}")
        return True
    except FileNotFoundError:
        print("❌ Error: seawater_model.pkl not found.")
        return False
    except Exception as e:
        print(f"❌ Error loading seawater model: {str(e)}")
        return False

def load_ice_model():
    """Load the ice melting rate model"""
    global ice_model_data
    
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
                        ice_model_data = pickle.load(f)
                else:
                    ice_model_data = joblib.load(model_path)
                print(f"✅ Ice model loaded successfully from: {model_path}")
                return True
            except Exception as e:
                print(f"❌ Error loading ice model from {model_path}: {str(e)}")
                continue
    
    print("❌ No ice model file found in any of the expected locations")
    return False

# ============= SEAWATER PREDICTION FUNCTIONS =============
def predict_seawater_properties(temperature_k: float, salinity_g_kg: float) -> Dict[str, float]:
    """Predict seawater properties given temperature and salinity"""
    if seawater_model_data is None:
        raise HTTPException(status_code=500, detail="Seawater model not loaded")
    
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
    input_scaled = seawater_model_data['scaler'].transform(input_enhanced)
    
    # Predict
    prediction = seawater_model_data['model'].predict(input_scaled)
    
    # Create result dictionary
    result = {}
    for i, target_name in enumerate(seawater_model_data['target_names']):
        result[target_name] = float(prediction[0][i])
    
    return result

# ============= ICE PREDICTION FUNCTIONS =============
def create_ice_features(temperature_k: float, year: int, month: int) -> np.ndarray:
    """Create feature array with the same engineering as training"""
    if ice_model_data is None:
        raise HTTPException(status_code=500, detail="Ice model not loaded")
    
    # Get normalization factors from model data
    year_min = ice_model_data['feature_engineering_info']['normalization_factors']['year_min']
    year_max = ice_model_data['feature_engineering_info']['normalization_factors']['year_max']
    
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
    """Predict ice melting rate using the loaded model"""
    if ice_model_data is None:
        raise HTTPException(status_code=500, detail="Ice model not loaded")
    
    try:
        # Create features
        features = create_ice_features(temperature_k, year, month)
        
        # Apply scaling if the model requires it
        if ice_model_data['uses_scaling'] and ice_model_data['scaler'] is not None:
            features = ice_model_data['scaler'].transform(features)
        
        # Make prediction
        prediction = ice_model_data['model'].predict(features)[0]
        return float(prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ice prediction error: {str(e)}")

# ============= API ENDPOINTS =============

# @app.on_event("startup")
# async def startup_event():
#     """Load models on startup"""
#     print("🚀 Loading models...")
#     seawater_success = load_seawater_model()
#     ice_success = load_ice_model()
    
#     if not seawater_success:
#         print("⚠️  Warning: Seawater model not loaded.")
#     if not ice_success:
#         print("⚠️  Warning: Ice model not loaded.")

# ============= UPDATE STARTUP EVENT =============
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("🚀 Loading models...")
    seawater_success = load_seawater_model()
    ice_success = load_ice_model()
    multi_output_success = load_multi_output_model()  # ADD THIS LINE
    
    if not seawater_success:
        print("⚠️  Warning: Seawater model not loaded.")
    if not ice_success:
        print("⚠️  Warning: Ice model not loaded.")
    if not multi_output_success:  # ADD THIS BLOCK
        print("⚠️  Warning: Multi-output model not loaded.")
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information"""
    return HealthResponse(
        status="healthy",
        message="Combined Environmental Prediction API is running",
        seawater_model_loaded=seawater_model_data is not None,
        ice_model_loaded=ice_model_data is not None
    )


# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint"""
#     seawater_loaded = seawater_model_data is not None
#     ice_loaded = ice_model_data is not None
    
#     if seawater_loaded and ice_loaded:
#         status = "healthy"
#         message = "Both models loaded successfully"
#     elif seawater_loaded or ice_loaded:
#         status = "degraded"
#         message = "Only one model loaded"
#     else:
#         status = "unhealthy"
#         message = "No models loaded"
    
#     return HealthResponse(
#         status=status,
#         message=message,
#         seawater_model_loaded=seawater_loaded,
#         ice_model_loaded=ice_loaded
#     )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    seawater_loaded = seawater_model_data is not None
    ice_loaded = ice_model_data is not None
    multi_output_loaded = multi_output_model_data is not None  # ADD THIS LINE
    
    loaded_count = sum([seawater_loaded, ice_loaded, multi_output_loaded])  # UPDATE THIS LINE
    
    if loaded_count == 3:  # CHANGE FROM 2 TO 3
        status = "healthy"
        message = "All models loaded successfully"
    elif loaded_count > 0:
        status = "degraded"
        message = f"{loaded_count}/3 models loaded"
    else:
        status = "unhealthy"
        message = "No models loaded"
    
    return {
        "status": status,
        "message": message,
        "seawater_model_loaded": seawater_loaded,
        "ice_model_loaded": ice_loaded,
        "multi_output_model_loaded": multi_output_loaded  # ADD THIS LINE
    }



@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded models"""
    seawater_info = None
    ice_info = None
    
    if seawater_model_data is not None:
        seawater_info = {
            "model_type": "Multi-Output Gradient Boosting Regressor",
            "feature_count": len(seawater_model_data['feature_names']),
            "target_count": len(seawater_model_data['target_names']),
            "feature_names": seawater_model_data['feature_names'],
            "target_names": seawater_model_data['target_names'],
            "test_r2_score": seawater_model_data['test_r2_score']
        }
    
    if ice_model_data is not None:
        ice_info = {
            "model_name": ice_model_data['model_name'],
            "test_r2_score": ice_model_data['performance']['test_r2_score'],
            "cv_score": ice_model_data['performance']['cv_score'],
            "rmse": ice_model_data['performance']['rmse'],
            "mae": ice_model_data['performance']['mae'],
            "features_used": ice_model_data['feature_columns'],
            "data_range": ice_model_data['data_stats']
        }
    
    return ModelInfo(seawater_model=seawater_info, ice_model=ice_info)


@app.post("/predict/multi-output", response_model=MultiOutputResponse)
async def predict_multi_output(input_data: MultiOutputInput):
    """
    Predict multiple output properties based on year, month, and temperature
    
    This endpoint uses the multi-output model to predict all target variables simultaneously.
    """
    if multi_output_model_data is None:
        raise HTTPException(status_code=503, detail="Multi-output model not loaded")
    
    try:
        # Make prediction
        predictions = predict_multi_output_properties(
            input_data.year,
            input_data.month, 
            input_data.temperature_k
        )
        
        # Prepare response
        response = MultiOutputResponse(
            input_parameters={
                "year": input_data.year,
                "month": input_data.month,
                "temperature_k": input_data.temperature_k,
                "temperature_celsius": input_data.temperature_k - 273.15
            },
            predictions=predictions,
            model_info={
                "model_name": multi_output_model_data['model_name'],
                "overall_r2_score": multi_output_model_data['performance']['overall_r2'],
                "overall_rmse": multi_output_model_data['performance']['overall_rmse'],
                "output_count": len(predictions),
                "output_features": multi_output_model_data['output_features']
            },
            prediction_timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-output prediction error: {str(e)}")

@app.post("/predict/multi-output-batch")
async def predict_multi_output_batch(input_list: list[MultiOutputInput]):
    """Batch prediction for multiple multi-output inputs"""
    if multi_output_model_data is None:
        raise HTTPException(status_code=503, detail="Multi-output model not loaded")
    
    if len(input_list) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")
    
    results = []
    for input_data in input_list:
        try:
            predictions = predict_multi_output_properties(
                input_data.year,
                input_data.month,
                input_data.temperature_k
            )
            
            results.append({
                "input": {
                    "year": input_data.year,
                    "month": input_data.month,
                    "temperature_k": input_data.temperature_k
                },
                "predictions": predictions,
                "success": True
            })
        except Exception as e:
            results.append({
                "input": {
                    "year": input_data.year,
                    "month": input_data.month,
                    "temperature_k": input_data.temperature_k
                },
                "error": str(e),
                "success": False
            })
    
    return {
        "batch_size": len(input_list),
        "results": results,
        "model_info": {
            "model_name": multi_output_model_data['model_name'],
            "overall_r2_score": multi_output_model_data['performance']['overall_r2']
        }
    }



# ============= SEAWATER ENDPOINTS =============
@app.post("/predict/seawater", response_model=SeawaterOutput)
async def predict_seawater(input_data: SeawaterInput):
    """
    Predict seawater properties based on temperature and salinity
    
    Returns predictions for all seawater properties including:
    - Density (kg/m3)
    - Ultrasonic Velocity (m/s)
    - Acoustic Impedance (Z) (kg/m2s) x104
    - Internal Pressure (πi) (MPa)
    - Heat Capacity (Cp)(kj/kgK)
    - Cohesion Energy Density (CED) (MPa)
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
                "r2_score": seawater_model_data['test_r2_score'],
                "prediction_count": len(predictions)
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seawater prediction error: {str(e)}")

@app.post("/predict/seawater-batch")
async def predict_seawater_batch(input_list: list[SeawaterInput]):
    """Predict seawater properties for multiple input combinations"""
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
            "r2_score": seawater_model_data['test_r2_score'] if seawater_model_data else None
        }
    }

# ============= ICE MELTING ENDPOINTS =============
@app.post("/predict/ice", response_model=IcePredictionResponse)
async def predict_ice(request: IcePredictionRequest):
    """Predict ice melting rate based on temperature, year, and month"""
    if ice_model_data is None:
        raise HTTPException(status_code=503, detail="Ice model not loaded")
    
    try:
        # Make prediction
        predicted_rate = predict_ice_melting_rate(
            request.temperature_k, 
            request.year, 
            request.month
        )
        
        return IcePredictionResponse(
            predicted_ice_melting_rate=round(predicted_rate, 2),
            temperature_k=request.temperature_k,
            year=request.year,
            month=request.month,
            model_used=ice_model_data['model_name'],
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/predict/ice")
async def predict_ice_get(
    temperature_k: float = Query(..., description="Temperature in Kelvin", ge=200.0, le=300.0),
    year: int = Query(..., description="Year", ge=1990, le=2030),
    month: int = Query(..., description="Month (1-12)", ge=1, le=12)
):
    """GET endpoint for ice prediction (alternative to POST)"""
    request = IcePredictionRequest(
        temperature_k=temperature_k,
        year=year,
        month=month
    )
    return await predict_ice(request)

@app.post("/predict/ice-batch")
async def predict_ice_batch(requests: list[IcePredictionRequest]):
    """Batch prediction for multiple ice melting inputs"""
    if ice_model_data is None:
        raise HTTPException(status_code=503, detail="Ice model not loaded")
    
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")
    
    results = []
    for req in requests:
        try:
            predicted_rate = predict_ice_melting_rate(req.temperature_k, req.year, req.month)
            results.append(IcePredictionResponse(
                predicted_ice_melting_rate=round(predicted_rate, 2),
                temperature_k=req.temperature_k,
                year=req.year,
                month=req.month,
                model_used=ice_model_data['model_name'],
                prediction_timestamp=datetime.now().isoformat()
            ))
        except Exception as e:
            results.append({
                "error": f"Failed to predict for temp={req.temperature_k}, year={req.year}, month={req.month}: {str(e)}"
            })
    
    return {"predictions": results}

# ============= UTILITY ENDPOINTS =============
# @app.post("/load-models")
# async def load_models_endpoint():
#     """Manually trigger model loading"""
#     seawater_success = load_seawater_model()
#     ice_success = load_ice_model()
    
#     return {
#         "seawater_model_loaded": seawater_success,
#         "ice_model_loaded": ice_success,
#         "message": f"Seawater: {'✅' if seawater_success else '❌'}, Ice: {'✅' if ice_success else '❌'}"
#     }

@app.post("/load-models")
async def load_models_endpoint():
    """Manually trigger model loading"""
    seawater_success = load_seawater_model()
    ice_success = load_ice_model()
    multi_output_success = load_multi_output_model()  # ADD THIS LINE
    
    return {
        "seawater_model_loaded": seawater_success,
        "ice_model_loaded": ice_success,
        "multi_output_model_loaded": multi_output_success,  # ADD THIS LINE
        "message": f"Seawater: {'✅' if seawater_success else '❌'}, Ice: {'✅' if ice_success else '❌'}, Multi-output: {'✅' if multi_output_success else '❌'}"  # UPDATE THIS LINE
    }

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

# if __name__ == "__main__":
#     print("🚀 Starting Combined Environmental Prediction API...")
#     print("📚 API Documentation will be available at: http://localhost:8000/docs")
#     print("🔍 Alternative docs at: http://localhost:8000/redoc")
#     print("\n🌊 Seawater endpoints:")
#     print("  - POST /predict/seawater")
#     print("  - POST /predict/seawater-batch")
#     print("\n🧊 Ice melting endpoints:")
#     print("  - POST /predict/ice")
#     print("  - GET /predict/ice")
#     print("  - POST /predict/ice-batch")
#     print("\n📊 General endpoints:")
#     print("  - GET /health")
#     print("  - GET /model-info")
if __name__ == "__main__":
    print("🚀 Starting Combined Environmental Prediction API...")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    print("\n🌊 Seawater endpoints:")
    print("  - POST /predict/seawater")
    print("  - POST /predict/seawater-batch")
    print("\n🧊 Ice melting endpoints:")
    print("  - POST /predict/ice")
    print("  - GET /predict/ice")
    print("  - POST /predict/ice-batch")
    print("\n🔮 Multi-output endpoints:")  # ADD THIS BLOCK
    print("  - POST /predict/multi-output")
    print("  - POST /predict/multi-output-batch")
    print("\n📊 General endpoints:")
    print("  - GET /health")
    print("  - GET /model-info")
    
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
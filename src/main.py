from pathlib import Path

from fastapi import FastAPI

from .schemas import (
    CompareOperatingConditionsRequest,
    DatasetSummaryResponse,
    DecayPrediction,
    HmiRulPredictionRequest,
    HmiRulPredictionResponse,
    HmiSurfaceMarkerRequest,
    MaintenanceRequest,
    MaintenanceResponse,
    PhysicalCorrelationsResponse,
    PredictFromSensorsRequest,
    PredictRequest,
    SensorAnalysisResponse,
)
from .service import (
    NavalPredictor,
    build_rul_prediction,
    build_hmi_snapshot,
    build_surface_data,
    build_surface_marker,
    maintenance_recommendation,
)

app = FastAPI(
    title="Naval Propulsion Condition Monitoring API",
    version="1.0.0",
    description="Backend API for decay prediction and maintenance decision support.",
)

DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "cleaned_data.csv"
predictor = NavalPredictor.from_csv(DATA_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/decay", response_model=DecayPrediction)
def predict_decay(payload: PredictRequest) -> DecayPrediction:
    return DecayPrediction(**predictor.predict_from_controls(payload.ship_speed, payload.lever_pos))


@app.post("/predict/decay-from-sensors", response_model=DecayPrediction)
def predict_decay_from_sensors(payload: PredictFromSensorsRequest) -> DecayPrediction:
    return DecayPrediction(**predictor.predict_from_sensors(payload.model_dump()))


@app.post("/analyze/sensors", response_model=SensorAnalysisResponse)
def analyze_sensors(payload: PredictFromSensorsRequest) -> SensorAnalysisResponse:
    return SensorAnalysisResponse(**predictor.sensor_analysis(payload.model_dump()))


@app.get("/dataset/summary", response_model=DatasetSummaryResponse)
def dataset_summary() -> DatasetSummaryResponse:
    return DatasetSummaryResponse(**predictor.dataset_summary())


@app.get("/correlations/physical", response_model=PhysicalCorrelationsResponse)
def physical_correlations() -> PhysicalCorrelationsResponse:
    return PhysicalCorrelationsResponse(**predictor.correlations())


@app.post("/maintenance/recommend", response_model=MaintenanceResponse)
def recommend_maintenance(payload: MaintenanceRequest) -> MaintenanceResponse:
    action, priority, components, window = maintenance_recommendation(
        payload.compressor_decay, payload.turbine_decay
    )
    return MaintenanceResponse(
        action=action,
        priority=priority,
        components=components,
        maintenance_window=window,
    )


@app.post("/tools/compare-operating-conditions")
def compare_operating_conditions(payload: CompareOperatingConditionsRequest) -> dict:
    return predictor.compare_operating_conditions(payload.speed_1, payload.speed_2)


@app.get("/hmi/snapshot")
def hmi_snapshot() -> dict:
    return build_hmi_snapshot(predictor)


@app.get("/hmi/surface-data")
def hmi_surface_data(speed: int) -> dict:
    return build_surface_data(predictor, speed)


@app.post("/hmi/surface-marker")
def hmi_surface_marker(payload: HmiSurfaceMarkerRequest) -> dict:
    return build_surface_marker(
        predictor,
        speed=payload.speed,
        compressor_decay_pred=payload.compressor_decay_pred,
        turbine_decay_pred=payload.turbine_decay_pred,
    )


@app.post("/hmi/rul-prediction", response_model=HmiRulPredictionResponse)
def hmi_rul_prediction(payload: HmiRulPredictionRequest) -> HmiRulPredictionResponse:
    return HmiRulPredictionResponse(
        **build_rul_prediction(
            predictor,
            ship_speed=payload.ship_speed,
            compressor_decay_pred=payload.compressor_decay_pred,
            turbine_decay_pred=payload.turbine_decay_pred,
        )
    )

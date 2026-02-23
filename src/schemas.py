from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Severity = Literal["critical", "warning", "caution", "healthy"]


class PredictRequest(BaseModel):
    ship_speed: int = Field(..., ge=3, le=27, description="Ship speed in knots")
    lever_pos: float = Field(..., ge=1.0, le=10.0, description="Lever position")


class PredictFromSensorsRequest(BaseModel):
    Lever_Pos: float
    Ship_Speed: float
    GT_Torque: float
    GT_RPM: float
    GG_RPM: float
    Prop_Torque_S: float
    Prop_Torque_P: float
    T48: float
    T2: float
    P48: float
    P2: float
    Pexh: float
    TIC: float
    Fuel_Flow: float


class DecayPrediction(BaseModel):
    compressor_decay: float
    turbine_decay: float
    severity: Severity
    operating_condition: str
    prediction_confidence: float = Field(
        ..., description="Reference model quality from report (R^2 approx.)"
    )


class SensorAnomaly(BaseModel):
    sensor: str
    value: float
    deviation: str
    normal_range: List[float]
    physical_meaning: str
    decay_correlation: Optional[float] = None
    affected_component: Literal["compressor", "turbine", "both", "unknown"]


class SensorAnalysisResponse(BaseModel):
    abnormal_sensors: List[SensorAnomaly]
    normal_sensors: int
    critical_anomalies: int


class DatasetSummaryResponse(BaseModel):
    total_operating_points: int
    speed_range: List[int]
    unique_speeds: List[int]
    critical_conditions: int
    warning_conditions: int
    avg_compressor_decay: float
    avg_turbine_decay: float
    worst_speed_condition: int
    healthiest_speed_condition: int


class CorrelationItem(BaseModel):
    correlation: float
    interpretation: str


class PhysicalCorrelationsResponse(BaseModel):
    temperature_correlations: Dict[str, CorrelationItem]
    pressure_correlations: Dict[str, CorrelationItem]
    efficiency_indicators: Dict[str, CorrelationItem]


class MaintenanceRequest(BaseModel):
    compressor_decay: float = Field(..., ge=0.9, le=1.0)
    turbine_decay: float = Field(..., ge=0.9, le=1.0)


class MaintenanceResponse(BaseModel):
    action: Literal["immediate", "scheduled", "monitor"]
    priority: Literal["high", "medium", "low"]
    components: List[str]
    maintenance_window: str


class CompareOperatingConditionsRequest(BaseModel):
    speed_1: int = Field(..., ge=3, le=27)
    speed_2: int = Field(..., ge=3, le=27)


class HmiSurfaceMarkerRequest(BaseModel):
    speed: int = Field(..., ge=3, le=27)
    compressor_decay_pred: float = Field(..., ge=0.9, le=1.0)
    turbine_decay_pred: float = Field(..., ge=0.9, le=1.0)


class HmiRulPredictionRequest(BaseModel):
    ship_speed: int = Field(..., ge=3, le=27)
    compressor_decay_pred: float = Field(..., ge=0.9, le=1.0)
    turbine_decay_pred: float = Field(..., ge=0.9, le=1.0)


class HmiRulPoint(BaseModel):
    unit: float
    decay: float


class HmiRulSeries(BaseModel):
    threshold: float
    current_decay: float
    slope_per_unit: float
    rul_units: int
    trend_basis: Literal["speed_specific", "global", "single_cycle"]
    points: List[HmiRulPoint]


class HmiRulPredictionResponse(BaseModel):
    ship_speed: int
    unit_label: str
    method: Literal["linear_projection"]
    compressor: HmiRulSeries
    turbine: HmiRulSeries
    next_maintenance: Dict[str, str | int]

export type Severity = "critical" | "warning" | "caution" | "healthy";

export interface DecayPrediction {
  compressor_decay: number;
  turbine_decay: number;
  severity: Severity;
  operating_condition: string;
  prediction_confidence: number;
}

export interface DatasetSummary {
  total_operating_points: number;
  speed_range: [number, number];
  unique_speeds: number[];
  critical_conditions: number;
  warning_conditions: number;
  avg_compressor_decay: number;
  avg_turbine_decay: number;
  worst_speed_condition: number;
  healthiest_speed_condition: number;
}

export interface CorrelationItem {
  correlation: number;
  interpretation: string;
}

export interface Correlations {
  temperature_correlations: Record<string, CorrelationItem>;
  pressure_correlations: Record<string, CorrelationItem>;
  efficiency_indicators: Record<string, CorrelationItem>;
}

export interface MaintenanceRecommendation {
  action: "immediate" | "scheduled" | "monitor";
  priority: "high" | "medium" | "low";
  components: string[];
  maintenance_window: string;
}

export interface BootstrapPayload {
  summary: DatasetSummary;
  correlations: Correlations;
  samplePrediction: DecayPrediction;
  recommendation: MaintenanceRecommendation;
}

export interface HmiSnapshot {
  snapshot_id: number;
  source: "csv_random_row" | "csv_holdout_row";
  split_meta?: {
    train_fraction: number;
    holdout_fraction: number;
    split_random_state: number;
    holdout_row_index: number;
  };
  operating_state: {
    ship_speed: number;
    lever_pos: number;
    gt_torque: number;
    gt_rpm: number;
    gg_rpm: number;
    tic: number;
    fuel_flow: number;
  };
  ship_state: {
    prop_torque_s: number;
    prop_torque_p: number;
  };
  temperature_state: {
    t1: number;
    t48: number;
    t2: number;
  };
  pressure_state: {
    p1: number;
    p48: number;
    p2: number;
    pexh: number;
  };
  predictions: {
    compressor_decay_pred: number;
    turbine_decay_pred: number;
    compressor_decay_actual: number;
    turbine_decay_actual: number;
    severity: "healthy" | "warning" | "critical";
    confidence_ref: number;
  };
  display_meta: {
    units: Record<string, string>;
    ranges: Record<string, [number, number]>;
    health_bands: Record<string, Record<string, string>>;
  };
}

export interface HmiSurfaceData {
  speed: number;
  axes: {
    turbine_decay_values: number[];
    compressor_decay_values: number[];
  };
  surface: {
    fuel_flow_z: number[][];
    t48_color: number[][];
  };
  plot_bounds: {
    x_min: number;
    x_max: number;
    y_min: number;
    y_max: number;
    z_min: number;
    z_max: number;
  };
  axis_meta: {
    invert_x: boolean;
    invert_y: boolean;
    x_label: string;
    y_label: string;
    z_label: string;
    color_label: string;
  };
}

export interface HmiSurfaceMarker {
  speed: number;
  marker: {
    turbine_decay: number;
    compressor_decay: number;
    fuel_flow: number;
    t48: number;
  };
}

export interface HmiRulPoint {
  unit: number;
  decay: number;
}

export interface HmiRulSeries {
  threshold: number;
  current_decay: number;
  slope_per_unit: number;
  rul_units: number;
  trend_basis: "speed_specific" | "global" | "single_cycle";
  points: HmiRulPoint[];
}

export interface HmiRulPrediction {
  ship_speed: number;
  unit_label: string;
  method: "linear_projection";
  compressor: HmiRulSeries;
  turbine: HmiRulSeries;
  next_maintenance: {
    component: "compressor" | "turbine";
    rul_units: number;
    status: "due_now" | "projected";
  };
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  toolTrace?: string[];
}

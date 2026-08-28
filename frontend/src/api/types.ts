// Mirrors backend/app/api/schemas.py. Optionals are optional here too: the API returns
// null rather than a placeholder whenever a real value does not exist.

export interface MappingProposal {
  source_column: string;
  normalized: string;
  role: "measurement" | "dimension" | "value_type" | "variable_name" | "value" | "unmapped";
  sample_values: string[];
  suggested_variable: string | null;
  suggested_value_type: string | null;
  confidence: number;
  ambiguity_gap: number;
  method: string | null;
  unit_conversion: string | null;
  decision: "auto_accept" | "needs_confirmation" | "confirmed" | "unmapped";
  alternatives: { variable: string; confidence: number }[];
}

export interface UploadResponse {
  batch_id: string;
  status: "pending_confirmation" | "training_started" | "failed";
  detected_format: string | null;
  layout: string | null;
  row_count_raw: number;
  row_count_ingested: number;
  skipped_rows: number;
  canonical_variables_found: string[];
  region_resolution_rate: number;
  source_profile_match: string;
  mapping_proposals: MappingProposal[];
  notes: string[];
}

export interface ConfirmMappingItem {
  source_column: string;
  variable?: string | null;
  value_type?: string | null;
  role?: string | null;
  unit_conversion?: string | null;
}

export interface RegionSummary {
  region_id: string;
  region_name: string | null;
  bust_probability: number | null;
  risk_band: RiskBand | null;
  confidence: number | null;
  dominant_variable: string | null;
  data_available: boolean;
}

export type RiskBand = "low" | "medium" | "high";

export interface RegionsResponse {
  lead_time_days: number;
  model_trained: boolean;
  last_trained_at: string | null;
  current_run_id: string | null;
  init_date: string | null;
  valid_date: string | null;
  risk_band_definitions: Record<string, string>;
  regions: RegionSummary[];
  message: string | null;
}

export interface VariablePoint {
  lead_time_days: number;
  valid_date: string | null;
  predicted_value: number | null;
  observed_value: number | null;
  predicted_error: number | null;
  confidence: number | null;
  ensemble_spread: number | null;
  ensemble_member_count: number | null;
}

export interface VariableSeries {
  variable: string;
  available: boolean;
  unit: string | null;
  bust_threshold: number | null;
  model_mae: number | null;
  model_rmse: number | null;
  model_r2: number | null;
  metrics_split: string | null;
  points: VariablePoint[];
}

export interface BustProbabilityPoint {
  lead_time_days: number;
  valid_date: string | null;
  bust_probability: number;
  risk_band: RiskBand;
  dominant_variable: string | null;
}

export interface TopFactor {
  feature: string;
  importance: number;
  method: string;
}

export interface RegionDetailResponse {
  region_id: string;
  region_name: string | null;
  model_trained: boolean;
  current_run_id: string | null;
  init_date: string | null;
  variables: VariableSeries[];
  bust_probability_curve: BustProbabilityPoint[];
  top_factors: TopFactor[];
  top_factors_method: string | null;
  analog_cases: unknown[];
  message: string | null;
}

export interface Alert {
  alert_id: string;
  region_id: string;
  region_name: string | null;
  lead_time_days: number;
  valid_date: string | null;
  bust_probability: number;
  risk_band: RiskBand;
  dominant_variable: string | null;
  created_at: string;
  training_run_id: string | null;
}

export interface AlertsResponse {
  generated_at: string;
  model_trained: boolean;
  risk_band_definitions: Record<string, string>;
  alerts: Alert[];
  message: string | null;
}

export interface ModelStatusResponse {
  model_trained: boolean;
  current_run_id: string | null;
  last_trained_at: string | null;
  training_in_progress: boolean;
  last_training_error: string | null;
  data_volume: {
    total_rows?: number;
    batches?: number;
    regions?: number;
    valid_date_min?: string | null;
    valid_date_max?: string | null;
    by_variable?: Record<string, Record<string, number>>;
    grain_counts?: Record<string, number>;
  };
  modelled_variables: string[];
  skipped_variables: Record<string, string>;
  validation_metrics: {
    regressors?: Record<string, {
      mae: number | null; rmse: number | null; r2: number | null;
      baseline_mae: number | null; split: string | null; n: number | null;
    }>;
    classifier?: Record<string, number | string | null>;
  };
  thresholds: {
    bust_threshold?: Record<string, number>;
    p90_error?: Record<string, number>;
    risk_band_cuts?: { medium: number; high: number };
    threshold_percentile?: number;
  };
  explanation_method: string | null;
  websocket_clients: number;
  message: string | null;
}

export type LiveEventType =
  | "connected"
  | "upload_received"
  | "mapping_pending"
  | "training_started"
  | "training_complete"
  | "training_failed"
  | "new_alert";

export interface LiveEvent {
  event: LiveEventType;
  timestamp: string;
  payload: Record<string, unknown>;
}

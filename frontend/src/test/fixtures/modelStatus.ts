import type { ModelStatusResponse } from "../../api/types";

/**
 * Trimmed from a real GET /api/model/status response against the live
 * deployment (https://sanket-a0dd.onrender.com), run run_20260916T050655Z,
 * fetched 2026-09-16. Numbers are copied verbatim, not invented; unrelated
 * top-level fields are still present at minimal/placeholder values only
 * because ModelStatusResponse marks them non-optional, not because the
 * baseline-ladder UI reads them.
 */
export const REAL_BASELINES_RUN_20260916: ModelStatusResponse["baselines"] = {
  run_id: "run_20260916T050655Z",
  test_events: 30490,
  test_cycles: 52,
  bust_rate: 0.48166612003935716,
  lead_bust_correlation: {
    train: 0.02803336794743909,
    test: -0.021551258127048594,
  },
  models: [
    { name: "climatology", brier: 0.2512843371974373, bss: 0.0, roc_auc: 0.5, f1: 0.0, is_model: false },
    {
      name: "lead_day", brier: 0.25182022970341816, bss: -0.0021326140417570194,
      roc_auc: 0.4877747026052116, f1: 0.0, is_model: false,
    },
    {
      name: "spread", brier: 0.24816894329685987, bss: 0.01239788335127956,
      roc_auc: 0.55726111000985, f1: 0.290289468281667, is_model: false,
    },
    {
      name: "lead+spread+season", brier: 0.2477305285247038, bss: 0.014142579328138671,
      roc_auc: 0.5623341236217476, f1: 0.3081698037645174, is_model: false,
    },
    {
      name: "Sanket bust classifier", brier: 0.18039620569514153, bss: 0.2821032631516467,
      roc_auc: 0.8074133821941691, f1: 0.7109798805191238, is_model: true,
    },
  ],
};

export const REAL_MODEL_STATUS_TRAINED: ModelStatusResponse = {
  model_trained: true,
  current_run_id: "run_20260916T050655Z",
  last_trained_at: "2026-09-16T05:22:05.410977Z",
  training_in_progress: false,
  last_training_error: null,
  data_volume: { total_rows: 1477376, regions: 71, forecast_cycles: 86 },
  training_data: {
    cycles: 350, train_cycles: 245, val_cycles: 53, held_out_cycles: 52,
    paired_rows: 4521073, first_train_date: "2000-01-09",
  },
  baselines: REAL_BASELINES_RUN_20260916,
  modelled_variables: [
    "atmospheric_moisture_kgm2", "humidity_pct", "pressure_hpa", "rainfall_mm",
    "soil_moisture_pct", "temperature_c", "wind_direction_deg", "wind_speed_ms",
  ],
  skipped_variables: {},
  validation_metrics: { regressors: {}, classifier: {} },
  thresholds: {},
  explanation_method: "shap",
  websocket_clients: 0,
  message: null,
};

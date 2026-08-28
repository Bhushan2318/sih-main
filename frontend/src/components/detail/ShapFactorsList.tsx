import type { TopFactor } from "../../api/types";

export function ShapFactorsList({ factors, method }: { factors: TopFactor[]; method: string | null }) {
  if (!factors.length) return <p className="muted">No explanation available for this region yet.</p>;
  const max = Math.max(...factors.map((f) => f.importance), 1e-9);

  return (
    <div>
      <ul className="factors">
        {factors.map((f) => (
          <li key={f.feature}>
            <span className="factors__name">{f.feature}</span>
            <span className="factors__bar" aria-hidden="true">
              <i style={{ width: `${(f.importance / max) * 100}%` }} />
            </span>
            <span className="factors__value">{f.importance.toFixed(4)}</span>
          </li>
        ))}
      </ul>
      {/* never let the UI imply SHAP ran when it didn't */}
      <p className="muted small">
        {method === "shap"
          ? "Mean |SHAP| on the validation split of the current run."
          : method === "feature_importance_fallback"
            ? "SHAP unavailable — showing XGBoost feature importances instead."
            : "Attribution method unknown."}
      </p>
    </div>
  );
}

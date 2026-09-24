import type { TopFactor } from "../../api/types";
import { featureLabel } from "../../lib/displayNames";

export function ShapFactorsList({ factors, method }: { factors: TopFactor[]; method: string | null }) {
  if (!factors.length) return <p className="muted">No explanation available for this region yet.</p>;
  const max = Math.max(...factors.map((f) => f.importance), 1e-9);

  return (
    <div>
      <ul className="factors">
        {factors.map((f) => (
          <li key={f.feature}>
            {/* The raw column name stays as the title: it is what the model calls this,
              * and anyone checking the feature list needs to be able to find it. */}
            <span className="factors__name" title={f.feature}>{featureLabel(f.feature)}</span>
            <span className="factors__bar" aria-hidden="true">
              <i style={{ width: `${(f.importance / max) * 100}%` }} />
            </span>
            <span className="factors__value">{f.importance.toFixed(4)}</span>
          </li>
        ))}
      </ul>
      <p className="muted small">
        {method === "shap"
          ? "How strongly each input moved this district's predictions, averaged over the " +
              "validation year (SHAP). A standing profile of the district, not a breakdown of " +
              "today's forecast alone."
          : method === "feature_importance_fallback"
            ? "SHAP unavailable — showing the model's own feature-importance ranking instead, " +
              "which reflects what it relies on overall rather than for this region."
            : "Attribution method unknown."}
      </p>
    </div>
  );
}

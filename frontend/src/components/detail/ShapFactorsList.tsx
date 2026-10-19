import { translate } from "../../i18n";
import type { TopFactor } from "../../api/types";
import { featureLabel } from "../../lib/displayNames";

export function ShapFactorsList({ factors, method }: { factors: TopFactor[]; method: string | null }) {
  if (!factors.length) return <p className="muted">{translate("shap.none")}</p>;
  const max = Math.max(...factors.map((f) => f.importance), 1e-9);

  return (
    <div>
      <ul className="factors">
        {factors.map((f) => (
          <li key={f.feature}>
            {/* The raw column name stays in the title: it is what the model calls this,
              * and anyone checking the feature list needs to be able to find it - but
              * labelled as such, not shown bare as if it were the name. */}
            <span className="factors__name" title={translate("shap.input", { name: f.feature })}>{featureLabel(f.feature)}</span>
            <span className="factors__bar" aria-hidden="true">
              <i style={{ width: `${(f.importance / max) * 100}%` }} />
            </span>
            <span className="factors__value">{f.importance.toFixed(4)}</span>
          </li>
        ))}
      </ul>
      <p className="muted small">
        {method === "shap"
          ? translate("shap.method")
          : method === "feature_importance_fallback"
            ? translate("shap.fallback")
            : translate("shap.unknown")}
      </p>
    </div>
  );
}

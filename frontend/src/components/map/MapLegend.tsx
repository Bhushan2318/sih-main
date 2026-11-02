import { translate } from "../../i18n";
import { bandLabel } from "../../theme";
import type { RiskBand, TextPart } from "../../api/types";
import { serverText } from "../../lib/serverText";

const BANDS: RiskBand[] = ["low", "medium", "high"];

export function MapLegend({ definitions = {}, parts = {} }: {
  definitions?: Record<string, string>;
  /** The same definitions as translatable parts; the English above is the fallback. */
  parts?: Record<string, TextPart>;
}) {
  const say = (k: string) => serverText(parts[k] ? [parts[k]] : undefined, definitions[k]);
  // The API's definitions are the source of truth for the explanatory text. Keep the
  // standard three swatches for a scored response, but do not invent low-risk semantics
  // when the backend returned no definitions (the usual shape of a no-score response).
  const describedBands = BANDS.filter((band) => Boolean(definitions[band]));

  return (
    <div className="legend">
      <span className="legend__title">{translate("legend.title")}</span>
      {describedBands.map((band) => (
        <span key={band} className="legend__item" title={say(band)}>
          <i className={`swatch swatch--${band}`} aria-hidden="true" />
          {bandLabel(band)}
        </span>
      ))}
      {describedBands.length === 0 ? (
        <span className="legend__item legend__item--muted">
          <i className="swatch swatch--nodata" aria-hidden="true" />
          {translate("legend.noBands")}
        </span>
      ) : null}
      {describedBands.length > 0 ? (
        <span className="legend__item legend__item--muted" title={translate("legend.noDataTitle")}>
          <i className="swatch swatch--nodata" aria-hidden="true" />
          {translate("legend.noData")}
        </span>
      ) : null}
      {definitions.basis ? <p className="legend__basis">{say("basis")}</p> : null}
    </div>
  );
}

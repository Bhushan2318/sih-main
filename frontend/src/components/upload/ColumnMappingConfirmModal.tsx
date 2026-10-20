import { translate } from "../../i18n";
import { useMemo, useState } from "react";
import type { ConfirmMappingItem, MappingProposal, UploadResponse } from "../../api/types";

const CANONICAL_VARIABLES = [
  "rainfall_mm",
  "temperature_c",
  "humidity_pct",
  "pressure_hpa",
  "atmospheric_moisture_kgm2",
  "soil_moisture_pct",
  "wind_speed_ms",
  "wind_direction_deg",
];

const VALUE_TYPES = ["forecast", "observed"];

interface Choice {
  variable: string;
  valueType: string;
}

export function ColumnMappingConfirmModal({ upload, onCancel, onSubmit }: {
  upload: UploadResponse;
  onCancel: () => void;
  onSubmit: (mappings: ConfirmMappingItem[]) => void;
}) {
  const needing = useMemo(
    () => upload.mapping_proposals.filter((p) => p.decision === "needs_confirmation"),
    [upload],
  );
  const auto = useMemo(
    () => upload.mapping_proposals.filter((p) => p.decision === "auto_accept" && p.role === "measurement"),
    [upload],
  );
  const excluded = useMemo(
    () => upload.mapping_proposals.filter((p) => p.role === "unmapped"),
    [upload],
  );

  const [choices, setChoices] = useState<Record<string, Choice>>(() =>
    Object.fromEntries(
      needing.map((p) => [
        p.source_column,
        { variable: p.suggested_variable ?? "", valueType: p.suggested_value_type ?? "" },
      ]),
    ),
  );

  const set = (col: string, patch: Partial<Choice>) =>
    setChoices((prev) => ({ ...prev, [col]: { ...prev[col], ...patch } }));

  const duplicates = useMemo(() => {
    const seen = new Map<string, number>();
    Object.values(choices).forEach((c) => {
      if (!c.variable) return;
      const key = `${c.variable}|${c.valueType}`;
      seen.set(key, (seen.get(key) ?? 0) + 1);
    });
    return new Set([...seen.entries()].filter(([, n]) => n > 1).map(([k]) => k));
  }, [choices]);

  const submit = () => {
    const mappings: ConfirmMappingItem[] = needing.map((p) => {
      const c = choices[p.source_column];
      if (!c?.variable) return { source_column: p.source_column, role: "unmapped" };
      return {
        source_column: p.source_column,
        variable: c.variable,
        value_type: c.valueType || null,
        unit_conversion: p.unit_conversion,
      };
    });
    onSubmit(mappings);
  };

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label={translate("map.title")}>
      <div className="modal">
        <header className="modal__head">
          <h3>{translate("map.title")}</h3>
          <button type="button" className="btn btn--ghost" onClick={onCancel} aria-label={translate("map.cancel")}>×</button>
        </header>

        <p className="muted small">
          {translate("map.meta", {
            fmt: upload.detected_format?.toUpperCase() ?? "",
            n: upload.row_count_raw.toLocaleString(),
            layout: upload.layout ?? translate("map.unknown"),
          })}
          {upload.source_profile_match !== "none" ? translate("map.matched", { match: upload.source_profile_match }) : ""}
        </p>

        {needing.length === 0 ? (
          <p>{translate("map.nothing")}</p>
        ) : (
          <div className="mapping-table" role="table">
            <div className="mapping-row mapping-row--head" role="row">
              <span>{translate("map.colColumn")}</span><span>{translate("map.colSamples")}</span><span>{translate("map.colVariable")}</span><span>{translate("map.colType")}</span><span>{translate("map.colConf")}</span>
            </div>
            {needing.map((p) => {
              const c = choices[p.source_column];
              const dupKey = `${c?.variable}|${c?.valueType}`;
              const isDup = Boolean(c?.variable) && duplicates.has(dupKey);
              return (
                <div className="mapping-row" role="row" key={p.source_column}>
                  <span className="mono">{p.source_column}</span>
                  <span className="muted small">{p.sample_values.slice(0, 3).join(", ")}</span>
                  <span>
                    <select value={c?.variable ?? ""} onChange={(e) => set(p.source_column, { variable: e.target.value })}>
                      <option value="">{translate("map.exclude")}</option>
                      {CANONICAL_VARIABLES.map((v) => <option key={v} value={v}>{v}</option>)}
                    </select>
                    {isDup ? <em className="warn small">{translate("map.dup")}</em> : null}
                    {p.unit_conversion ? <em className="muted small"> ({p.unit_conversion})</em> : null}
                  </span>
                  <span>
                    <select
                      value={c?.valueType ?? ""}
                      disabled={!c?.variable}
                      onChange={(e) => set(p.source_column, { valueType: e.target.value })}
                    >
                      <option value="">{translate("map.fromData")}</option>
                      {VALUE_TYPES.map((v) => <option key={v} value={v}>{v}</option>)}
                    </select>
                  </span>
                  <span className="mono small">{(p.confidence * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        )}

        {auto.length ? (
          <details>
            <summary className="muted small">{translate("map.auto", { n: auto.length })}</summary>
            <ul className="notes">
              {auto.map((p: MappingProposal) => (
                <li key={p.source_column}>{p.source_column} → {p.suggested_variable}</li>
              ))}
            </ul>
          </details>
        ) : null}

        {excluded.length ? (
          <details>
            <summary className="muted small">{translate("map.excluded", { n: excluded.length })}</summary>
            <ul className="notes">{excluded.map((p) => <li key={p.source_column}>{p.source_column}</li>)}</ul>
          </details>
        ) : null}

        <footer className="modal__foot">
          {duplicates.size ? (
            <span className="warn small">
              {translate("map.dupWarn")}
            </span>
          ) : null}
          <button type="button" className="btn btn--ghost" onClick={onCancel}>{translate("map.cancel")}</button>
          <button
            type="button"
            className="btn btn--primary"
            onClick={submit}
            disabled={duplicates.size > 0}
          >
            {translate("map.confirm")}
          </button>
        </footer>
      </div>
    </div>
  );
}

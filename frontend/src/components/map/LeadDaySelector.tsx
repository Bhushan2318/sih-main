import { translate } from "../../i18n";

const DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

export function LeadDaySelector({ value, onChange, disabled, available }: {
  value: number;
  onChange: (d: number) => void;
  disabled?: boolean;
  available?: number[];
}) {
  const covered = (d: number) => !available?.length || available.includes(d);
  return (
    <div className="lead-selector" role="group" aria-label={translate("lead.group")}>
      <span className="lead-selector__label">{translate("lead.label")}</span>
      {DAYS.map((d) => {
        const missing = !covered(d);
        return (
          <button
            key={d}
            type="button"
            disabled={disabled || missing}
            aria-pressed={d === value}
            title={missing ? translate("lead.missing", { d }) : translate("lead.day", { d })}
            className={
              d === value ? "lead-btn lead-btn--active"
                : missing ? "lead-btn lead-btn--missing" : "lead-btn"
            }
            onClick={() => onChange(d)}
          >
            {d}
          </button>
        );
      })}
    </div>
  );
}

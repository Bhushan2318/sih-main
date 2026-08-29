const DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/**
 * Lead-day picker. Days the current cycle does not cover are disabled rather than hidden,
 * so the horizon stays legible: a 06/12/18 UTC run cannot produce a whole-calendar-day
 * forecast for its own init day, so it has no day 1.
 */
export function LeadDaySelector({ value, onChange, disabled, available }: {
  value: number;
  onChange: (d: number) => void;
  disabled?: boolean;
  available?: number[];
}) {
  const covered = (d: number) => !available?.length || available.includes(d);
  return (
    <div className="lead-selector" role="group" aria-label="Forecast lead day">
      <span className="lead-selector__label">Lead day</span>
      {DAYS.map((d) => {
        const missing = !covered(d);
        return (
          <button
            key={d}
            type="button"
            disabled={disabled || missing}
            aria-pressed={d === value}
            title={missing ? `This cycle has no day-${d} forecast` : `Lead day ${d}`}
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

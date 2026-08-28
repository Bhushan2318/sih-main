const DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

export function LeadDaySelector({ value, onChange, disabled }: {
  value: number;
  onChange: (d: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className="lead-selector" role="group" aria-label="Forecast lead day">
      <span className="lead-selector__label">Lead day</span>
      {DAYS.map((d) => (
        <button
          key={d}
          type="button"
          disabled={disabled}
          aria-pressed={d === value}
          className={d === value ? "lead-btn lead-btn--active" : "lead-btn"}
          onClick={() => onChange(d)}
        >
          {d}
        </button>
      ))}
    </div>
  );
}

import type { ReplayCycleSummary } from "../../api/types";
import { formatDateUtc } from "../../lib/format";

/** What a past event is, which kind of unseen year it comes from, and what the model was
 * allowed to know. The title is the only hand-written text; the sample note comes from the
 * run that scored the event. */
export function ReplayEventHeader({ cycle }: { cycle: ReplayCycleSummary }) {
  return (
    <div className="replay__event">
      <strong className="replay__event-title">{cycle.title}</strong>
      {cycle.sample_note ? <span className="replay__event-note">{cycle.sample_note}</span> : null}
      <span className="muted small">
        Scored with what was known on {formatDateUtc(cycle.init_date)}, 00 UTC · checked against ERA5
      </span>
    </div>
  );
}

import type { ReplayCycleSummary, ReplayFocusSeries, ReplayLeadStep } from "../../api/types";
import { eventHeadline } from "../../lib/eventHeadline";
import { formatDateUtc } from "../../lib/format";

/** What a past event is, which kind of unseen year it comes from, and what the model was
 * allowed to know. The title is the only hand-written text; the sample note and the
 * headline sentence are both read off the scored cycle. */
export function ReplayEventHeader({
  cycle,
  steps,
  focusOptions,
}: {
  cycle: ReplayCycleSummary;
  steps: ReplayLeadStep[];
  focusOptions: ReplayFocusSeries[];
}) {
  const headline = eventHeadline(cycle, steps, focusOptions);
  return (
    <div className="replay__event">
      <strong className="replay__event-title">{cycle.title}</strong>
      {cycle.sample_note ? <span className="replay__event-note">{cycle.sample_note}</span> : null}
      <span className="muted small">
        Scored with what was known on {formatDateUtc(cycle.init_date)}, 00 UTC · checked against ERA5
      </span>
      {headline ? <p className="replay__event-headline">{headline}</p> : null}
    </div>
  );
}

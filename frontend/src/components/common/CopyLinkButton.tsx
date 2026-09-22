import { useEffect, useRef, useState } from "react";

/**
 * Copies the current address, which now carries the tab, district, lead day and filter.
 *
 * The deep links are useless if nobody knows they exist: an address bar that quietly
 * updates is invisible to someone being shown the site for the first time. This is the
 * affordance that says the thing on screen has an address - "look at Chennai on day 5"
 * becomes a link rather than four clicks of instructions.
 *
 * Falls back to selecting the text when the clipboard is unavailable. `navigator
 * .clipboard` needs a secure context, and it can also be refused by permissions policy,
 * so it is never assumed - the button reports what actually happened rather than
 * claiming success.
 */
export function CopyLinkButton({ label = "Copy link" }: { label?: string }) {
  const [said, setSaid] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => () => {
    if (timer.current) window.clearTimeout(timer.current);
  }, []);

  const announce = (text: string) => {
    setSaid(text);
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setSaid(null), 2200);
  };

  const copy = async () => {
    const url = window.location.href;
    try {
      if (!navigator.clipboard) throw new Error("no clipboard");
      await navigator.clipboard.writeText(url);
      announce("Link copied");
    } catch {
      // Not a failure worth a dialog: show the address so it can be copied by hand.
      window.prompt("Copy this link", url);
      setSaid(null);
    }
  };

  return (
    <>
      {/* aria-label pins the accessible name, so the visible text can change to report
        * the result without the button renaming itself under a screen reader - and
        * without the status below being read out twice. */}
      <button
        type="button"
        className="chip"
        onClick={copy}
        aria-label={label}
        title="Copy a link to exactly this view"
      >
        {said ?? label}
      </button>
      {/* Announced politely, so the result is heard without stealing focus. */}
      <span className="visually-hidden" role="status" aria-live="polite">{said ?? ""}</span>
    </>
  );
}

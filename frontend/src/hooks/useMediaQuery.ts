import { useEffect, useState } from "react";

/**
 * Subscribe to a media query from React.
 *
 * Used where a breakpoint has to change *what renders*, not just how it looks: two copies
 * of the same control hidden from each other by CSS would put two tablists in the
 * accessibility tree and let focus land on the invisible one.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = () => setMatches(mq.matches);
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

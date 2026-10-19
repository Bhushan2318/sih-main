import { cloneElement, Fragment, type ReactElement, type ReactNode } from "react";
import { translate, type Key } from "./index";

/**
 * A translated sentence with inline markup. The catalogue holds the whole sentence with
 * numbered tags - "<1>{{n}}</1> in the bust band" - so a translator (or the model) can
 * move the emphasised words to wherever that language's grammar puts them. Tag N is
 * rendered as a clone of `parts[N]` wrapping the tagged text. Tags do not nest.
 *
 * Splitting the sentence into separately translated fragments instead would produce word
 * order that is right in English and wrong in most of the 22 languages.
 */
export function rich(key: Key, params: Record<string, string | number> | undefined, parts: ReactElement[]): ReactNode {
  return richText(translate(key, params), parts);
}

export function richText(text: string, parts: ReactElement[]): ReactNode {
  const out: ReactNode[] = [];
  const re = /<(\d+)>(.*?)<\/\1>/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const el = parts[Number(m[1])];
    out.push(el ? cloneElement(el, { key: out.length }, m[2]) : m[2]);
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return <Fragment>{out}</Fragment>;
}

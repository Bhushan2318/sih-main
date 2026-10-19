import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { richText } from "./rich";

describe("richText", () => {
  it("wraps numbered tags in the given elements, wherever the sentence puts them", () => {
    const { container } = render(<p>{richText("of 9, <1>4</1> busts", [<i />, <b className="x" />])}</p>);
    expect(container.innerHTML).toBe('<p>of 9, <b class="x">4</b> busts</p>');
  });

  it("leaves text alone when a tag has no element", () => {
    const { container } = render(<p>{richText("<3>a</3> b", [])}</p>);
    expect(container.textContent).toBe("a b");
  });
});

describe("every shipped string renders clean", () => {
  const files = import.meta.glob<Record<string, string>>("./locales/*.json", { eager: true, import: "default" });
  for (const [path, cat] of Object.entries(files)) {
    it(`${path}: no raw {{params}} or <N> tags reach the screen`, () => {
      for (const [k, v] of Object.entries(cat)) {
        if (k.startsWith("_")) continue;
        const filled = v.replace(/\{\{(\w+)\}\}/g, "7");
        const { container, unmount } = render(<p>{richText(filled, [<b />, <i />, <a />, <span />])}</p>);
        expect(container.textContent, `${path} ${k}`).not.toMatch(/[{}]|<\/?\d+>/);
        unmount();
      }
    });
  }
});

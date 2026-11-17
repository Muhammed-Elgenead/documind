import { describe, expect, it } from "vitest";

import { reflow } from "./format";

describe("reflow", () => {
  it("joins layout line breaks and keeps paragraphs", () => {
    expect(reflow("Up to 5 days may be\ncarried over.\n\nNew para\nhere.")).toBe(
      "Up to 5 days may be carried over.\n\nNew para here.",
    );
  });

  it("rejoins words hyphenated across lines", () => {
    expect(reflow("Full-\ntime employees")).toBe("Full-time employees");
  });

  it("normalizes CRLF and trims", () => {
    expect(reflow("  a\r\nb  \r\n\r\n\r\nc ")).toBe("a b\n\nc");
  });
});

import { describe, expect, it } from "vitest";

import { citationNumber, linkifyCitations } from "./citations";

describe("linkifyCitations", () => {
  it("links single and grouped markers", () => {
    expect(linkifyCitations("Revenue grew [1]. Costs rose [2, 3].")).toBe(
      "Revenue grew [1](#cite-1). Costs rose [2](#cite-2)[3](#cite-3).",
    );
  });

  it("links adjacent markers", () => {
    expect(linkifyCitations("Claim [1][3].")).toBe("Claim [1](#cite-1)[3](#cite-3).");
  });

  it("leaves real links and fenced code alone", () => {
    const md = "See [1](https://x.y) and\n```\narr[0] = list[2]\n```\nok [4]";
    expect(linkifyCitations(md)).toBe(
      "See [1](https://x.y) and\n```\narr[0] = list[2]\n```\nok [4](#cite-4)",
    );
  });
});

describe("citationNumber", () => {
  it("extracts the number from chip links only", () => {
    expect(citationNumber("#cite-7")).toBe(7);
    expect(citationNumber("https://example.com")).toBeNull();
    expect(citationNumber("#cite-x")).toBeNull();
    expect(citationNumber(undefined)).toBeNull();
  });
});

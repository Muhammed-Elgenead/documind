// End-to-end check of the whole flow in a real browser:
//   sign up -> upload (drag-and-drop) -> live status -> ask -> streamed answer -> click citation -> panel.
//
// Needs the frontend (npm run dev) and backend running. Usage:
//   node e2e/flow.mjs <file.pdf> "<question>" [--base http://localhost:5173] [--shots dir]
import fs from "node:fs";
import path from "node:path";

import { chromium, devices } from "playwright";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args.splice(i, 2)[1] : fallback;
};
const BASE = opt("--base", "http://localhost:5173");
const SHOTS = opt("--shots", "e2e/screenshots");
const [pdfPath, question] = args;
if (!pdfPath || !question) {
  console.error('Usage: node e2e/flow.mjs <file.pdf> "<question>" [--base url] [--shots dir]');
  process.exit(2);
}
fs.mkdirSync(SHOTS, { recursive: true });
const filename = path.basename(pdfPath);
const step = (msg) => console.log(`✓ ${msg}`);

const PASSWORD = "e2e-password-123";

async function signUp(page, email) {
  await page.getByTestId("auth-form").waitFor();
  await page.getByTestId("auth-switch").click();
  await page.getByTestId("email-input").fill(email);
  await page.getByTestId("password-input").fill(PASSWORD);
  await page.getByTestId("auth-submit").click();
  await page.getByTestId("user-email").filter({ hasText: email }).waitFor();
}

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1360, height: 860 } });

  // 0. Signed out, every page leads to the sign-in form; sign up, land where we were going.
  const email = `e2e-${Date.now()}@example.com`;
  await page.goto(`${BASE}/documents`);
  await page.waitForURL(/\/login$/);
  step("signed out: /documents redirects to the sign-in page");
  await signUp(page, email);
  await page.waitForURL(/\/documents$/);
  step(`signed up as ${email}; back on /documents`);

  // 1. Upload by dropping a file on the dropzone (a real drop event with DataTransfer).
  const b64 = fs.readFileSync(pdfPath).toString("base64");
  const dataTransfer = await page.evaluateHandle(
    ({ b64, name }) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const dt = new DataTransfer();
      dt.items.add(new File([bytes], name, { type: "application/pdf" }));
      return dt;
    },
    { b64, name: filename },
  );
  await page.dispatchEvent('[data-testid="dropzone"]', "drop", { dataTransfer });
  const row = page.getByTestId("document-row").filter({ hasText: filename }).first();
  await row.waitFor();
  const firstStatus = await row.getByTestId("doc-status").innerText();
  step(`dropped ${filename}; status shows "${firstStatus}"`);
  await page.screenshot({ path: `${SHOTS}/1-uploading.png` });

  // 2. Live status: the list polls until the document is ready.
  await row.getByTestId("doc-status").filter({ hasText: "ready" }).waitFor({ timeout: 300_000 });
  step(`status became "ready" without reloading (${await row.locator("p").nth(1).innerText()})`);
  await page.screenshot({ path: `${SHOTS}/2-ready.png` });

  // 3. Ask a question; watch the answer stream in.
  await page.getByRole("link", { name: /ask a question/i }).click();
  await page.getByTestId("question-input").fill(question);
  await page.getByTestId("send-button").click();
  const answer = page.getByTestId("assistant-message").last();
  const lengths = new Set();
  const t0 = Date.now();
  while (Date.now() - t0 < 180_000) {
    lengths.add((await answer.innerText()).length);
    if (await answer.getByTestId("citation-chip").count()) break;
    await page.waitForTimeout(100);
  }
  const chips = answer.getByTestId("citation-chip");
  if (!(await chips.count())) throw new Error("No clickable citation chips appeared.");
  step(`answer streamed in (${lengths.size} distinct partial lengths seen)`);
  await page.waitForURL(/\/c\/[0-9a-f-]{36}$/);
  step(`URL is now ${new URL(page.url()).pathname}`);
  const sidebar = page.getByTestId("conversation-list");
  await sidebar.getByText(question.slice(0, 20)).first().waitFor();
  step("conversation appears in the sidebar");
  await page.screenshot({ path: `${SHOTS}/3-answer.png` });

  // 4. Click a citation chip -> side panel with filename, page and exact passage.
  const chip = chips.first();
  const label = await chip.getAttribute("aria-label");
  await chip.click();
  const panel = page.getByTestId("citation-panel");
  await panel.waitFor();
  const shownFile = await panel.getByTestId("citation-filename").innerText();
  const shownPage = await panel.getByTestId("citation-page").innerText();
  const passage = await panel.getByTestId("citation-text").innerText();
  if (shownFile !== filename) throw new Error(`Panel shows ${shownFile}, expected ${filename}`);
  if (!/^Page \d+$/.test(shownPage)) throw new Error(`Unexpected page label: ${shownPage}`);
  if (passage.trim().length < 20) throw new Error("Panel passage is empty.");
  step(`clicked chip (${label}); panel shows ${shownFile}, ${shownPage}`);
  step(`passage: "${passage.slice(0, 90).replace(/\s+/g, " ")}…"`);
  await page.screenshot({ path: `${SHOTS}/4-citation-panel.png` });
  await page.keyboard.press("Escape");
  await panel.waitFor({ state: "detached" });
  step("Escape closes the panel");

  // 5. Reload: the conversation and its citations come back from the database.
  await page.reload();
  const restored = page.getByTestId("assistant-message").last().getByTestId("citation-chip").first();
  await restored.click();
  await page.getByTestId("citation-panel").waitFor();
  step("after reload, saved answer and citations still work");

  // 6. Phone-sized viewport: panel becomes a bottom sheet.
  const token = await page.evaluate(() => localStorage.getItem("documind.token"));
  const mobile = await browser.newPage({ ...devices["iPhone 13"] });
  await mobile.addInitScript((t) => localStorage.setItem("documind.token", t), token);
  await mobile.goto(page.url());
  await mobile.getByTestId("assistant-message").last().getByTestId("citation-chip").first().click();
  await mobile.getByTestId("citation-panel").waitFor();
  await mobile.screenshot({ path: `${SHOTS}/5-mobile-panel.png` });
  step("mobile layout: citation opens as a bottom sheet");
  await mobile.close();

  // 7. Another account sees none of it.
  const conversationUrl = page.url();
  await page.getByTestId("sign-out").click();
  await page.waitForURL(/\/login$/);
  await signUp(page, `e2e-other-${Date.now()}@example.com`);
  await page.getByTestId("conversation-list").getByText("No conversations yet").waitFor();
  await page.goto(`${BASE}/documents`);
  await page.getByText(/no documents/i).first().waitFor();
  if (await page.getByTestId("document-row").count()) throw new Error("Second user sees documents.");
  await page.goto(conversationUrl);
  await page.getByText(/not found/i).first().waitFor();
  step("a second account sees no documents or conversations, and can't open the first one's chat");

  console.log(`\nAll steps passed. Screenshots in ${SHOTS}/`);
} catch (err) {
  console.error(`\n✗ ${err.message}`);
  process.exitCode = 1;
} finally {
  await browser.close();
}

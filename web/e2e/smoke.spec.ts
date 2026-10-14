import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import path from "path";
import fs from "fs";

const SPEECH_ID = "speech-001";
const AUDIO_PATH = path.resolve(
  "D:/ECHO (Evaluative Contrastive Holistic Orations)/speechlens/data/demo/speech-001_clean_L0.flac"
);
const TRANSCRIPT =
  "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live.";

test.describe("SpeechLens smoke test", () => {
  test("dashboard loads and health check passes", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("header")).toContainText("SpeechLens");
  });

  test("health endpoint returns status ok", async ({ request }) => {
    const res = await request.get("/api/health");
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body.status).toBe("ok");
  });

  test("speeches list endpoint returns array", async ({ request }) => {
    const res = await request.get("/api/speeches");
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(Array.isArray(body)).toBeTruthy();
    expect(body.length).toBeGreaterThan(0);
  });

  test("demo button loads precomputed result and renders full visualisations", async ({ page }) => {
    await page.goto("/");

    // Wait for demo clips to appear
    await expect(page.locator("text=Demo Clips:")).toBeVisible({ timeout: 10000 });
    const firstDemoBtn = page.locator("button", { hasText: /Ideal Baseline|Level 1/i }).first();
    await expect(firstDemoBtn).toBeVisible({ timeout: 10000 });
    await firstDemoBtn.click();

    // Verify score panel renders
    const scoreHeading = page.locator("text=Speech Performance");
    await expect(scoreHeading).toBeVisible({ timeout: 15000 });
    await expect(page.locator("text=/ 100")).toBeVisible();

    // Verify transcript strip renders
    await expect(page.getByRole("heading", { name: "Transcript" })).toBeVisible();

    // Verify overlay chart renders
    await expect(page.locator("text=Pitch · Energy · Tempo")).toBeVisible();
  });

  test("accessibility check has no critical violations on demo dashboard", async ({ page }) => {
    await page.goto("/");
    // Click demo button to load full UI
    const firstDemoBtn = page.locator("button", { hasText: /Ideal Baseline|Level 1/i }).first();
    await expect(firstDemoBtn).toBeVisible({ timeout: 10000 });
    await firstDemoBtn.click();
    await expect(page.locator("text=Speech Performance")).toBeVisible({ timeout: 15000 });

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .analyze();

    const criticalViolations = results.violations.filter(
      (v) => v.impact === "critical"
    );
    expect(criticalViolations).toEqual([]);
  });

  test("upload library clip renders score panel", async ({ page }) => {
    test.setTimeout(120000);

    if (!fs.existsSync(AUDIO_PATH)) {
      test.skip(true, "Library audio not available");
      return;
    }

    await page.goto("/");

    // Upload the file via native filechooser
    const fileChooserPromise = page.waitForEvent("filechooser");
    await page.getByText("Click to upload or drag audio file").click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(AUDIO_PATH);

    // Verify audio file selection is detected
    await expect(page.locator("text=Selected:")).toBeVisible({ timeout: 10000 });

    // Click Analyze
    const analyzeBtn = page.locator("button", { hasText: /Analyze/i });
    await expect(analyzeBtn).toBeEnabled({ timeout: 5000 });
    await analyzeBtn.click();

    // Wait for the score panel to appear
    const scoreHeading = page.locator("text=Speech Performance");
    await expect(scoreHeading).toBeVisible({ timeout: 90000 });

    // Verify a score value is visible
    const scoreValue = page.locator("text=/ 100");
    await expect(scoreValue).toBeVisible({ timeout: 5000 });
  });
});

import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 120000,
  retries: 0,
  use: {
    baseURL: "http://localhost:8000",
    ...devices["Desktop Chrome"],
    headless: true,
  },
  webServer: {
    command: "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000",
    cwd: "D:/ECHO (Evaluative Contrastive Holistic Orations)/speechlens",
    url: "http://localhost:8000/api/health",
    reuseExistingServer: false,
    timeout: 30000,
  },
});

import { processMindatGalleryPage } from "./mindat_browser_download_helper.mjs";

const hostProcess = process;
const [metadataBase64, mode, rootDir, mineralKey, pageText] = hostProcess.argv.slice(2);

if (!metadataBase64 || !mode) {
  throw new Error("Usage: node mindat_browser_runner.mjs <metadata-base64> <inspect|page> [root] [mineral] [page]");
}

const requestMeta = JSON.parse(Buffer.from(metadataBase64, "base64url").toString("utf8"));
globalThis.nodeRepl = {
  requestMeta,
  env: {},
  homeDir: hostProcess.env.USERPROFILE || "C:\\Users\\bob",
  tmpDir: hostProcess.env.TEMP || "C:\\Temp",
  setResponseMeta() {},
  async emitImage() {},
};

const { setupBrowserRuntime } = await import(
  "file:///C:/Users/bob/.codex/plugins/cache/openai-bundled/browser/26.707.71524/scripts/browser-client.mjs"
);

await setupBrowserRuntime({ globals: globalThis });
const browser = await agent.browsers.getForUrl("https://www.mindat.org/gm/1572");

if (mode === "inspect") {
  console.log(await browser.documentation());
  console.log(JSON.stringify(await browser.tabs.list(), null, 2));
  hostProcess.exit(0);
}

if (mode !== "page" || !rootDir || !mineralKey || !/^\d+$/.test(pageText || "")) {
  throw new Error("Page mode requires <root> <mineral> <page>.");
}

const tab = await browser.tabs.getActive();
const result = await processMindatGalleryPage(tab, Number(pageText), rootDir, mineralKey);
console.log(JSON.stringify(result, null, 2));

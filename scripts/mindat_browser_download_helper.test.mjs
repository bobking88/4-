import assert from "node:assert/strict";
import test from "node:test";

import { MINERAL_CONFIGS, waitForMatchingImageAssets } from "./mindat_browser_download_helper.mjs";
import { loadGalleryLazyImages } from "./mindat_browser_download_helper.mjs";
import { readGalleryEntriesAfterLazyLoad } from "./mindat_browser_download_helper.mjs";
import { waitForDownloadablePhotoEntries } from "./mindat_browser_download_helper.mjs";

test("fluorapatite uses the dedicated Mindat gallery and metadata naming", () => {
  assert.deepEqual(MINERAL_CONFIGS.fluorapatite, {
    galleryId: 1572,
    incomingFolder: "fluorapatite",
    logPrefix: "mindat_fluorapatite",
    filenamePrefix: "fluorapatite",
    mineralLabel: "fluorapatite",
    exactTitle: "Fluorapatite",
  });
});

test("waitForMatchingImageAssets waits until requested gallery images are registered", async () => {
  let calls = 0;
  let pauses = 0;
  const assetsCap = {
    async list() {
      calls += 1;
      return {
        id: "inventory-1",
        assets:
          calls === 1
            ? []
            : calls === 2
              ? [{ id: "asset-1", kind: "image", url: "https://example.test/a.jpg" }]
              : [
                  { id: "asset-1", kind: "image", url: "https://example.test/a.jpg" },
                  { id: "asset-2", kind: "image", url: "https://example.test/b.jpg" },
                ],
      };
    },
  };
  const tab = {
    playwright: {
      async waitForTimeout() {
        pauses += 1;
      },
    },
  };

  const result = await waitForMatchingImageAssets(
    tab,
    assetsCap,
    new Set(["https://example.test/a.jpg", "https://example.test/b.jpg"]),
    { timeoutMs: 1000, pollMs: 1 },
  );

  assert.equal(result.assets.length, 2);
  assert.equal(calls, 3);
  assert.equal(pauses, 2);
});

test("loadGalleryLazyImages asks the browser page to scroll through lazy-loaded thumbnails", async () => {
  const evaluations = [];
  let pauses = 0;
  const tab = {
    playwright: {
      async evaluate(fn, arg) {
        evaluations.push({ fn: String(fn), arg });
        if (evaluations.length === 1) return { maxScroll: 1200, step: 600 };
      },
      async waitForTimeout() {
        pauses += 1;
      },
    },
  };

  await loadGalleryLazyImages(tab);

  assert.equal(evaluations.length, 4);
  assert.match(evaluations[0].fn, /scrollHeight/);
  assert.match(evaluations[1].fn, /scrollTo/);
  assert.deepEqual(evaluations.slice(1).map((call) => call.arg), [0, 600, undefined]);
  assert.equal(pauses, 2);
});

test("readGalleryEntriesAfterLazyLoad collects thumbnails only after scrolling the gallery", async () => {
  const calls = [];
  const tab = {
    playwright: {
      async evaluate(fn, arg) {
        const source = String(fn);
        calls.push({ source, arg });
        if (source.includes("scrollHeight")) return { maxScroll: 600, step: 600 };
        if (source.includes("document.images")) return [{ pid: "42", src: "https://example.test/a.jpg" }];
      },
      async waitForTimeout() {},
    },
  };

  const entries = await readGalleryEntriesAfterLazyLoad(tab);

  assert.equal(entries.length, 1);
  assert.match(calls.at(-1).source, /document\.images/);
  assert.match(calls[0].source, /scrollHeight/);
});

test("waitForDownloadablePhotoEntries chooses the asset-backed thumbnail for each photo page", async () => {
  const candidates = [
    { href: "https://example.test/photo-1.html", src: "https://example.test/placeholder-1.jpg" },
    { href: "https://example.test/photo-1.html", src: "https://example.test/photo-1.jpg" },
    { href: "https://example.test/photo-2.html", src: "https://example.test/placeholder-2.jpg" },
    { href: "https://example.test/photo-2.html", src: "https://example.test/photo-2.jpg" },
  ];
  const assetsCap = {
    async list() {
      return {
        id: "inventory-1",
        assets: [
          { id: "asset-1", kind: "image", url: "https://example.test/photo-1.jpg" },
          { id: "asset-2", kind: "image", url: "https://example.test/photo-2.jpg" },
        ],
      };
    },
  };
  const tab = { playwright: { async waitForTimeout() {} } };

  const result = await waitForDownloadablePhotoEntries(tab, assetsCap, candidates);

  assert.deepEqual(result.entries.map((entry) => entry.src), [
    "https://example.test/photo-1.jpg",
    "https://example.test/photo-2.jpg",
  ]);
  assert.equal(result.assets.length, 2);
});

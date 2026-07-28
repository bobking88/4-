import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

export const MINERAL_CONFIGS = {
  magnetite: {
    galleryId: 2538,
    incomingFolder: "magnetite",
    logPrefix: "mindat_magnetite",
    filenamePrefix: "magnetite",
    mineralLabel: "magnetite",
    exactTitle: "Magnetite",
  },
  ilmenite: {
    galleryId: 2013,
    incomingFolder: "ilmenite",
    logPrefix: "mindat_ilmenite",
    filenamePrefix: "ilmenite",
    mineralLabel: "ilmenite",
    exactTitle: "Ilmenite",
  },
  titanomagnetite: {
    galleryId: 3978,
    incomingFolder: "titanomagnetite",
    logPrefix: "mindat_titanomagnetite",
    filenamePrefix: "titanomagnetite",
    mineralLabel: "titanomagnetite",
    exactTitle: "Magnetite (Var: Titanium-bearing Magnetite)",
  },
  perovskite: {
    galleryId: 3166,
    incomingFolder: "perovskite",
    logPrefix: "mindat_perovskite",
    filenamePrefix: "perovskite",
    mineralLabel: "perovskite",
    exactTitle: "Perovskite",
  },
  rutile: {
    galleryId: 3486,
    incomingFolder: "rutile",
    logPrefix: "mindat_rutile",
    filenamePrefix: "rutile",
    mineralLabel: "rutile",
    exactTitle: "Rutile",
  },
  anatase: {
    galleryId: 213,
    incomingFolder: "anatase",
    logPrefix: "mindat_anatase",
    filenamePrefix: "anatase",
    mineralLabel: "anatase",
    exactTitle: "Anatase",
  },
  titanite: {
    galleryId: 3977,
    incomingFolder: "titanite",
    logPrefix: "mindat_titanite",
    filenamePrefix: "titanite",
    mineralLabel: "titanite",
    exactTitle: "Titanite",
  },
  fluorapatite: {
    galleryId: 1572,
    incomingFolder: "fluorapatite",
    logPrefix: "mindat_fluorapatite",
    filenamePrefix: "fluorapatite",
    mineralLabel: "fluorapatite",
    exactTitle: "Fluorapatite",
  },
  vesuvianite: {
    galleryId: 4223,
    incomingFolder: "vesuvianite",
    logPrefix: "mindat_vesuvianite",
    filenamePrefix: "vesuvianite",
    mineralLabel: "vesuvianite",
    exactTitle: "Vesuvianite",
  },
  pyrite: {
    galleryId: 3314,
    incomingFolder: "pyrite",
    logPrefix: "mindat_pyrite",
    filenamePrefix: "pyrite",
    mineralLabel: "pyrite",
    exactTitle: "Pyrite",
  },
  hematite: {
    galleryId: 1856,
    incomingFolder: "hematite",
    logPrefix: "mindat_hematite",
    filenamePrefix: "hematite",
    mineralLabel: "hematite",
    exactTitle: "Hematite",
  },
  goethite: {
    galleryId: 1719,
    incomingFolder: "goethite",
    logPrefix: "mindat_goethite",
    filenamePrefix: "goethite",
    mineralLabel: "goethite",
    exactTitle: "Goethite",
  },
  chalcopyrite: {
    galleryId: 955,
    incomingFolder: "chalcopyrite",
    logPrefix: "mindat_chalcopyrite",
    filenamePrefix: "chalcopyrite",
    mineralLabel: "chalcopyrite",
    exactTitle: "Chalcopyrite",
  },
  quartz: {
    galleryId: 3337,
    incomingFolder: "quartz",
    logPrefix: "mindat_quartz",
    filenamePrefix: "quartz",
    mineralLabel: "quartz",
    exactTitle: "Quartz",
  },
  feldspar: {
    galleryId: 1624,
    incomingFolder: "feldspar",
    logPrefix: "mindat_feldspar",
    filenamePrefix: "feldspar",
    mineralLabel: "feldspar",
    exactTitle: "Feldspar Group",
  },
  calcite: {
    galleryId: 859,
    incomingFolder: "calcite",
    logPrefix: "mindat_calcite",
    filenamePrefix: "calcite",
    mineralLabel: "calcite",
    exactTitle: "Calcite",
  },
  pyroxene: {
    galleryId: 9767,
    incomingFolder: "pyroxene",
    logPrefix: "mindat_pyroxene",
    filenamePrefix: "pyroxene",
    mineralLabel: "pyroxene",
    exactTitle: "Pyroxene Group",
  },
};

function csvEscape(value) {
  const s = String(value ?? "");
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function linesOf(text) {
  return String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function creditOf(text) {
  return linesOf(text).find((line) => /copyright|photo|collection|specimen/i.test(line)) || "";
}

function isChinaLocality(locality) {
  return /\bChina\b|中国/.test(String(locality || ""));
}

function screeningDecision(title, exactTitle) {
  const text = String(title || "").trim();
  const pattern = new RegExp(`^${exactTitle}(?!\\s*,)`, "i");
  return pattern.test(text) ? "keep_prelim" : "mixed_uncertain_review";
}

export async function waitForMatchingImageAssets(
  tab,
  assetsCap,
  wantedUrls,
  { timeoutMs = 15000, pollMs = 1000 } = {},
) {
  const startedAt = Date.now();
  let inventory = null;
  let assets = [];

  while (Date.now() - startedAt <= timeoutMs) {
    inventory = await assetsCap.list();
    assets = inventory.assets.filter((asset) => asset.kind === "image" && wantedUrls.has(asset.url));
    if (assets.length === wantedUrls.size) return { inventory, assets };
    if (Date.now() - startedAt >= timeoutMs) break;
    await tab.playwright.waitForTimeout(pollMs);
  }

  return { inventory, assets };
}

export async function waitForDownloadablePhotoEntries(
  tab,
  assetsCap,
  candidates,
  { timeoutMs = 15000, pollMs = 1000 } = {},
) {
  const groups = new Map();
  for (const candidate of candidates) {
    const group = groups.get(candidate.href) || [];
    group.push(candidate);
    groups.set(candidate.href, group);
  }

  const startedAt = Date.now();
  let inventory = null;
  let entries = [];
  let assets = [];
  while (Date.now() - startedAt <= timeoutMs) {
    inventory = await assetsCap.list();
    const assetByUrl = new Map(
      inventory.assets
        .filter((asset) => asset.kind === "image")
        .map((asset) => [asset.url, asset]),
    );
    entries = [];
    assets = [];
    for (const group of groups.values()) {
      const entry = group.find((candidate) => assetByUrl.has(candidate.src));
      if (entry) {
        entries.push(entry);
        assets.push(assetByUrl.get(entry.src));
      }
    }
    if (entries.length === groups.size) return { inventory, entries, assets };
    if (Date.now() - startedAt >= timeoutMs) break;
    await tab.playwright.waitForTimeout(pollMs);
  }

  return { inventory, entries, assets };
}

export async function loadGalleryLazyImages(tab) {
  const { maxScroll, step } = await tab.playwright.evaluate(() => ({
    maxScroll: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
    step: Math.max(
      Math.ceil(Math.max(document.body.scrollHeight, document.documentElement.scrollHeight) / 5),
      Math.floor(window.innerHeight * 0.8),
      600,
    ),
  }));
  for (let top = 0; top < maxScroll; top += step) {
    await tab.playwright.evaluate((scrollTop) => window.scrollTo(0, scrollTop), top);
    await tab.playwright.waitForTimeout(75);
  }
  await tab.playwright.evaluate(() => window.scrollTo(0, 0));
}

export async function readGalleryEntriesAfterLazyLoad(tab) {
  await loadGalleryLazyImages(tab);
  return tab.playwright.evaluate(
    () => {
      const out = [];
      for (const img of Array.from(document.images)) {
        const src = img.currentSrc || img.src || "";
        const anchor = img.closest('a[href*="photo-"]');
        const href = anchor?.href || "";
        if (!src || !href || !/photo-\d+\.html/i.test(href) || !/imagecache/i.test(src)) continue;
        const text = (anchor?.innerText || "").trim();
        const lines = text
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean);
        out.push({
          pid: (href.match(/photo-(\d+)\.html/i) || [])[1] || "",
          href,
          src,
          title: lines[0] || "",
          locality: lines[1] || "",
          text,
          width: img.naturalWidth || 0,
          height: img.naturalHeight || 0,
        });
      }
      return out;
    },
    undefined,
    { timeoutMs: 30000 },
  );
}

export async function processMindatGalleryPage(tab, pageNo, rootDir, mineralKey = "magnetite") {
  const config = MINERAL_CONFIGS[mineralKey];
  if (!config) {
    throw new Error(`Unsupported Mindat mineral key: ${mineralKey}`);
  }

  const incomingDir = path.join(rootDir, "incoming_downloads", config.incomingFolder);
  const metadataDir = path.join(rootDir, "metadata");
  await fs.mkdir(incomingDir, { recursive: true });
  await fs.mkdir(metadataDir, { recursive: true });

  try {
    await tab.goto(`https://www.mindat.org/gm/${config.galleryId}?page=${pageNo}`);
  } catch {
    // Mindat sometimes completes rendering even when browser navigation reports a timeout.
  }
  await tab.playwright.waitForTimeout(3000);

  const currentUrl = await tab.url();
  const candidates = await readGalleryEntriesAfterLazyLoad(tab);
  const screeningEntries = Array.from(new Map(candidates.map((entry) => [entry.href, entry])).values());

  const chinaCount = screeningEntries.filter((entry) => isChinaLocality(entry.locality)).length;
  if (chinaCount >= Math.max(3, Math.ceil(screeningEntries.length * 0.5))) {
    const skipLog = path.join(metadataDir, `${config.logPrefix}_page${pageNo}_SKIPPED_CHINA.txt`);
    await fs.writeFile(
      skipLog,
      `Skipped page ${pageNo}: detected China localities in ${chinaCount}/${entries.length}.\nURL: ${currentUrl}\n`,
      "utf8",
    );
    return {
      pageNo,
      url: currentUrl,
      entries: screeningEntries.length,
      skipped: true,
      reason: "China page skipped",
      skipLog,
    };
  }

  const assetsCap = await tab.capabilities.get("pageAssets");
  const { inventory, entries, assets } = await waitForDownloadablePhotoEntries(tab, assetsCap, candidates);
  if (!inventory || entries.length !== screeningEntries.length) {
    throw new Error(
      `Mindat gallery image assets did not become available for every ${config.exactTitle} photo on page ${pageNo} within 15 seconds.`,
    );
  }
  const bundle = await assetsCap.bundle({
    inventoryId: inventory.id,
    assetIds: Array.from(new Map(assets.map((asset) => [asset.id, asset])).values()).map((asset) => asset.id),
  });
  const assetByUrl = new Map(bundle.assets.map((asset) => [asset.url, asset]));

  const headers = [
    "source_filename",
    "mineral_label",
    "mindat_photo_id",
    "detail_page_url",
    "download_source_url",
    "windows_referrer_url",
    "windows_host_url",
    "page_title",
    "locality",
    "photographer_or_credit",
    "license_or_rights",
    "screening_decision",
    "notes",
  ];

  const rows = [];
  const failures = [];
  for (let i = 0; i < entries.length; i += 1) {
    const entry = entries[i];
    const asset = assetByUrl.get(entry.src);
    if (!asset) {
      failures.push({ pid: entry.pid, error: "asset not bundled" });
      continue;
    }
    const pid = entry.pid || String(i + 1).padStart(6, "0");
    const filename = `${config.filenamePrefix}_mindat_p${String(pid).padStart(6, "0")}_page${pageNo}_${String(i + 1).padStart(3, "0")}_raw.jpg`;
    const destination = path.join(incomingDir, filename);
    try {
      await fs.copyFile(asset.path, destination);
      const bytes = await fs.readFile(destination);
      const sha = crypto.createHash("sha256").update(bytes).digest("hex");
      rows.push({
        source_filename: `incoming_downloads/${config.incomingFolder}/${filename}`,
        mineral_label: config.mineralLabel,
        mindat_photo_id: pid,
        detail_page_url: entry.href,
        download_source_url: entry.src,
        windows_referrer_url: entry.href,
        windows_host_url: entry.src,
        page_title: entry.title || config.exactTitle,
        locality: entry.locality,
        photographer_or_credit: creditOf(entry.text),
        license_or_rights: "Mindat photo page rights; verify per photo before republication",
        screening_decision: screeningDecision(entry.title, config.exactTitle),
        notes: `Downloaded via browser pageAssets from Mindat ${config.exactTitle} gallery page ${pageNo}; image_sha256=${sha}; description=${entry.text}`,
      });
    } catch (error) {
      failures.push({ pid, error: String(error?.message || error) });
    }
  }

  const csv =
    headers.join(",") +
    "\n" +
    rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")).join("\n") +
    "\n";
  const logPath = path.join(metadataDir, `${config.logPrefix}_page${pageNo}_download_log.csv`);
  await fs.writeFile(logPath, csv, "utf8");

  return {
    pageNo,
    url: currentUrl,
    mineralKey,
    entries: entries.length,
    chinaCount,
    assetsMatched: assets.length,
    copied: rows.length,
    failed: failures.length,
    skipped: false,
    logPath,
    bundleFailures: bundle.failures.length,
    failures: failures.slice(0, 5),
  };
}

export async function processVisibleMindatGalleryBatch(
  tab,
  pageNo,
  rootDir,
  mineralKey,
  { offset = 0, limit = 6, navigate = true } = {},
) {
  const config = MINERAL_CONFIGS[mineralKey];
  if (!config) throw new Error(`Unsupported Mindat mineral key: ${mineralKey}`);

  const incomingDir = path.join(rootDir, "incoming_downloads", config.incomingFolder);
  const metadataDir = path.join(rootDir, "metadata");
  await fs.mkdir(incomingDir, { recursive: true });
  await fs.mkdir(metadataDir, { recursive: true });

  if (navigate) {
    try {
      await tab.goto(`https://www.mindat.org/gm/${config.galleryId}?page=${pageNo}`);
    } catch {
      // Continue because Mindat can finish rendering after navigation times out.
    }
  }

  const currentUrl = await tab.url();
  const candidates = await tab.playwright.evaluate(() => {
    const out = [];
    for (const img of Array.from(document.images)) {
      const src = img.currentSrc || img.src || "";
      const anchor = img.closest('a[href*="photo-"]');
      const href = anchor?.href || "";
      if (!src || !href || !/photo-\d+\.html/i.test(href) || !/imagecache/i.test(src)) continue;
      const lines = (anchor?.innerText || "")
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
      out.push({
        pid: (href.match(/photo-(\d+)\.html/i) || [])[1] || "",
        href,
        src,
        title: lines[0] || "",
        locality: lines[1] || "",
        text: lines.join("\n"),
      });
    }
    return out;
  });
  const uniqueEntries = Array.from(new Map(candidates.map((entry) => [entry.href, entry])).values());
  const batchCandidates = uniqueEntries.slice(offset, offset + limit);
  if (batchCandidates.length === 0) {
    throw new Error(`No visible ${config.exactTitle} gallery entries found on page ${pageNo}.`);
  }

  const assetsCap = await tab.capabilities.get("pageAssets");
  const { inventory, entries, assets } = await waitForDownloadablePhotoEntries(tab, assetsCap, batchCandidates, {
    timeoutMs: 8000,
    pollMs: 500,
  });
  if (!inventory || entries.length !== batchCandidates.length) {
    throw new Error(`Visible Mindat image assets were not ready for batch ${offset}-${offset + batchCandidates.length - 1}.`);
  }
  const bundle = await assetsCap.bundle({
    inventoryId: inventory.id,
    assetIds: assets.map((asset) => asset.id),
  });
  const assetByUrl = new Map(bundle.assets.map((asset) => [asset.url, asset]));

  const headers = [
    "source_filename",
    "mineral_label",
    "mindat_photo_id",
    "detail_page_url",
    "download_source_url",
    "windows_referrer_url",
    "windows_host_url",
    "page_title",
    "locality",
    "photographer_or_credit",
    "license_or_rights",
    "screening_decision",
    "notes",
  ];
  const rows = [];
  const failures = [];
  for (let i = 0; i < entries.length; i += 1) {
    const entry = entries[i];
    const asset = assetByUrl.get(entry.src);
    const pagePosition = offset + i + 1;
    const pid = entry.pid || String(pagePosition).padStart(6, "0");
    const filename = `${config.filenamePrefix}_mindat_p${String(pid).padStart(6, "0")}_page${pageNo}_${String(pagePosition).padStart(3, "0")}_raw.jpg`;
    try {
      if (!asset) throw new Error("asset not bundled");
      const destination = path.join(incomingDir, filename);
      await fs.copyFile(asset.path, destination);
      const sha = crypto.createHash("sha256").update(await fs.readFile(destination)).digest("hex");
      rows.push({
        source_filename: `incoming_downloads/${config.incomingFolder}/${filename}`,
        mineral_label: config.mineralLabel,
        mindat_photo_id: pid,
        detail_page_url: entry.href,
        download_source_url: entry.src,
        windows_referrer_url: entry.href,
        windows_host_url: entry.src,
        page_title: entry.title || config.exactTitle,
        locality: entry.locality,
        photographer_or_credit: creditOf(entry.text),
        license_or_rights: "Mindat photo page rights; verify per photo before republication",
        screening_decision: screeningDecision(entry.title, config.exactTitle),
        notes: `Downloaded via visible browser batch from Mindat ${config.exactTitle} gallery page ${pageNo}; image_sha256=${sha}; description=${entry.text}`,
      });
    } catch (error) {
      failures.push({ pid, error: String(error?.message || error) });
    }
  }

  const csv = `${headers.join(",")}\n${rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")).join("\n")}\n`;
  const logPath = path.join(metadataDir, `${config.logPrefix}_page${pageNo}_batch${offset + 1}-${offset + batchCandidates.length}_download_log.csv`);
  await fs.writeFile(logPath, csv, "utf8");
  return {
    pageNo,
    offset,
    requested: batchCandidates.length,
    copied: rows.length,
    failed: failures.length,
    visibleEntries: uniqueEntries.length,
    url: currentUrl,
    logPath,
    failures,
  };
}

export async function processMindatMagnetitePage(tab, pageNo, rootDir) {
  return processMindatGalleryPage(tab, pageNo, rootDir, "magnetite");
}

export async function processMindatIlmenitePage(tab, pageNo, rootDir) {
  return processMindatGalleryPage(tab, pageNo, rootDir, "ilmenite");
}

export async function processMindatTitanomagnetitePage(tab, pageNo, rootDir) {
  return processMindatGalleryPage(tab, pageNo, rootDir, "titanomagnetite");
}

export async function processMindatPerovskitePage(tab, pageNo, rootDir) {
  return processMindatGalleryPage(tab, pageNo, rootDir, "perovskite");
}

export async function processMindatRutilePage(tab, pageNo, rootDir) {
  return processMindatGalleryPage(tab, pageNo, rootDir, "rutile");
}

export async function processMindatAnatasePage(tab, pageNo, rootDir) {
  return processMindatGalleryPage(tab, pageNo, rootDir, "anatase");
}

export async function processMindatTitanitePage(tab, pageNo, rootDir) {
  return processMindatGalleryPage(tab, pageNo, rootDir, "titanite");
}

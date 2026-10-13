#!/usr/bin/env node
// A static file server with the right MIME types, no dependencies. `--isolate` adds the
// COOP/COEP headers that enable SharedArrayBuffer; kevala works without them.
//
// usage: node scripts/serve.mjs [directory] [--port 8080] [--isolate] [--cors]
import { createReadStream, lstatSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, relative, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { MODELS } from "../js/src/source.js";
import { PREVIEW_MODEL_IDS } from "../app/model-policy.js";
import { discoverLocalPacks } from "./lib/local-packs.mjs";
import { resolveRequestPath } from "./lib/static-path.mjs";

const types = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm", ".svg": "image/svg+xml",
  ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp", ".woff2": "font/woff2", ".kevala": "application/octet-stream",
  ".md": "text/markdown; charset=utf-8",
};
const previewModels = Object.fromEntries(Object.entries(MODELS).filter(([id]) => PREVIEW_MODEL_IDS.has(id)));

function baseHeaders(path, isolate, cors) {
  const headers = { "content-type": types[extname(path)] || "application/octet-stream", "cache-control": "no-cache" };
  if (isolate) Object.assign(headers, { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" });
  if (cors) Object.assign(headers, { "access-control-allow-origin": "*", "access-control-expose-headers": "content-length, content-range", "cross-origin-resource-policy": "cross-origin" });
  return headers;
}

function notFound(res) {
  return res.writeHead(404, { "content-type": "text/plain; charset=utf-8", "cache-control": "no-cache" }).end("not found");
}

function sendStream(req, res, path, range) {
  const stream = createReadStream(path, range ? { start: range.start, end: range.end } : undefined);
  const stop = () => stream.destroy();
  const cleanup = () => {
    req.off("aborted", stop);
    res.off("close", stop);
  };
  req.once("aborted", stop);
  res.once("close", stop);
  stream.once("close", cleanup);
  stream.once("error", () => {
    cleanup();
    if (res.destroyed) return;
    if (!res.headersSent) res.writeHead(500, { "content-type": "text/plain; charset=utf-8" }).end("could not read file");
    else res.destroy();
  });
  stream.pipe(res);
}

function parseRange(value, size) {
  const match = /^bytes=(\d*)-(\d*)$/i.exec(value.trim());
  if (!match || (!match[1] && !match[2]) || size <= 0) return { invalid: true };

  const bound = (digits) => {
    const normalized = digits.replace(/^0+(?=\d)/, "");
    const sizeText = String(size);
    const exceeds = normalized.length > sizeText.length
      || (normalized.length === sizeText.length && normalized > sizeText);
    return { exceeds, value: exceeds ? size : Number(normalized) };
  };

  if (!match[1]) {
    const suffix = bound(match[2]);
    if (!suffix.value) return { invalid: true };
    return { start: suffix.exceeds ? 0 : Math.max(0, size - suffix.value), end: size - 1 };
  }

  const start = bound(match[1]);
  const requestedEnd = match[2] ? bound(match[2]) : { exceeds: false, value: size - 1 };
  if (start.exceeds || start.value >= size || (!requestedEnd.exceeds && requestedEnd.value < start.value)) return { invalid: true };
  return { start: start.value, end: requestedEnd.exceeds ? size - 1 : Math.min(requestedEnd.value, size - 1) };
}

function sendFile(req, res, path, st, headers) {
  const rangeValue = req.headers.range;
  if (rangeValue) {
    const range = parseRange(rangeValue, st.size);
    if (range.invalid) {
      return res.writeHead(416, { ...headers, "content-range": `bytes */${st.size}`, "content-length": 0, "accept-ranges": "bytes" }).end();
    }
    const length = range.end - range.start + 1;
    res.writeHead(206, {
      ...headers,
      "content-range": `bytes ${range.start}-${range.end}/${st.size}`,
      "content-length": length,
      "accept-ranges": "bytes",
    });
    if (req.method !== "HEAD") sendStream(req, res, path, range);
    else res.end();
    return;
  }

  res.writeHead(200, { ...headers, "content-length": st.size, "accept-ranges": "bytes" });
  if (req.method !== "HEAD" && st.size > 0) sendStream(req, res, path, null);
  else res.end();
}

/** Create the local static server. The optional catalog lets integration tests use tiny fixtures. */
export function createKevalaServer({ root = ".", isolate = false, cors = false, models = previewModels } = {}) {
  const serveRoot = resolve(root);

  return createServer((req, res) => {
    const pathname = new URL(req.url || "/", "http://x").pathname;
    const resolved = resolveRequestPath(serveRoot, pathname);
    if (resolved.status) return res.writeHead(resolved.status).end();

    const relativePath = relative(serveRoot, resolved.path).split(sep).join("/");
    if (req.method === "OPTIONS") {
      return res.writeHead(204, {
        ...baseHeaders(resolved.path, isolate, cors),
        "access-control-allow-headers": "range",
      }).end();
    }

    if (relativePath === ".kevala/models.json") {
      const { manifest } = discoverLocalPacks(serveRoot, models);
      if (!Object.keys(manifest.models).length) return notFound(res);
      const body = Buffer.from(JSON.stringify(manifest));
      const headers = { ...baseHeaders("models.json", isolate, cors), "content-length": body.length };
      res.writeHead(200, headers);
      return req.method === "HEAD" ? res.end() : res.end(body);
    }

    if (relativePath.startsWith(".kevala/")) {
      const inventory = discoverLocalPacks(serveRoot, models);
      const pack = inventory.files.get(`/${relativePath}`);
      if (!pack) return notFound(res);
      let st;
      try {
        st = lstatSync(pack.path);
      } catch {
        return notFound(res);
      }
      if (!st.isFile() || st.size !== pack.bytes) return notFound(res);
      return sendFile(req, res, pack.path, st, baseHeaders(pack.path, isolate, cors));
    }

    let path = resolved.path;
    let st;
    try {
      st = statSync(path);
      if (st.isDirectory()) {
        path = join(path, "index.html");
        st = statSync(path);
      }
    } catch {
      return notFound(res);
    }
    if (!st.isFile()) return notFound(res);
    return sendFile(req, res, path, st, baseHeaders(path, isolate, cors));
  });
}

function main(argv) {
  const { values: options, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: { port: { type: "string", default: "8080" }, isolate: { type: "boolean", default: false }, cors: { type: "boolean", default: false } },
  });
  if (positionals.length > 1) throw new Error("serve one directory");
  const root = resolve(positionals[0] || ".");
  const port = Number(options.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error(`--port must be a port number, not ${options.port}`);
  createKevalaServer({ root, isolate: options.isolate, cors: options.cors }).listen(port, "127.0.0.1", () => {
    console.log(`serving ${root} on http://127.0.0.1:${port}${options.isolate ? " (cross-origin isolated)" : ""}`);
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) main(process.argv.slice(2));

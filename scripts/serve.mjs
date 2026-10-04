#!/usr/bin/env node
// A static file server with the right MIME types, no dependencies. `--isolate` adds the
// COOP/COEP headers that enable SharedArrayBuffer; kevala works without them.
//
// usage: node scripts/serve.mjs [directory] [--port 8080] [--isolate] [--cors]
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { resolveRequestPath } from "./lib/static-path.mjs";

const { values: options, positionals } = parseArgs({
  allowPositionals: true,
  options: { port: { type: "string", default: "8080" }, isolate: { type: "boolean", default: false }, cors: { type: "boolean", default: false } },
});
if (positionals.length > 1) throw new Error("serve one directory");
const root = resolve(positionals[0] || ".");
const port = Number(options.port);
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error(`--port must be a port number, not ${options.port}`);
const isolate = options.isolate;
// --cors: answer like a CDN (Access-Control-Allow-Origin: *), to test cross-origin embedding
const cors = options.cors;
const types = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm", ".svg": "image/svg+xml",
  ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp", ".woff2": "font/woff2", ".kevala": "application/octet-stream",
  ".md": "text/markdown; charset=utf-8",
};

createServer((req, res) => {
  const resolved = resolveRequestPath(root, new URL(req.url, "http://x").pathname);
  if (resolved.status) return res.writeHead(resolved.status).end();
  let { path } = resolved;
  let st;
  try {
    st = statSync(path);
    if (st.isDirectory()) {
      path = join(path, "index.html");
      st = statSync(path);
    }
  } catch {
    return res.writeHead(404, { "content-type": "text/plain" }).end("not found");
  }
  const headers = { "content-type": types[extname(path)] || "application/octet-stream", "cache-control": "no-cache" };
  if (isolate) Object.assign(headers, { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" });
  if (cors) Object.assign(headers, { "access-control-allow-origin": "*", "access-control-expose-headers": "content-length, content-range", "cross-origin-resource-policy": "cross-origin" });
  if (req.method === "OPTIONS") return res.writeHead(204, { ...headers, "access-control-allow-headers": "range" }).end();
  // byte ranges, so large packs can be probed without downloading them
  const range = req.headers.range?.match(/bytes=(\d+)-(\d*)/);
  if (range) {
    const start = Number(range[1]);
    const end = range[2] ? Number(range[2]) : st.size - 1;
    res.writeHead(206, { ...headers, "content-range": `bytes ${start}-${end}/${st.size}`, "content-length": end - start + 1, "accept-ranges": "bytes" });
    return createReadStream(path, { start, end }).pipe(res);
  }
  res.writeHead(200, { ...headers, "content-length": st.size, "accept-ranges": "bytes" });
  createReadStream(path).pipe(res);
}).listen(port, "127.0.0.1", () => console.log(`serving ${root} on http://127.0.0.1:${port}${isolate ? " (cross-origin isolated)" : ""}`));

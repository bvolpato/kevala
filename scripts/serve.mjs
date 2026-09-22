#!/usr/bin/env node
// A static file server with the right MIME types, no dependencies. `--isolate` adds the
// COOP/COEP headers that enable SharedArrayBuffer; kevala works without them.
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";

const args = process.argv.slice(2);
const root = resolve(args.find((a) => !a.startsWith("--")) || ".");
const port = Number(args.find((a) => a.startsWith("--port="))?.slice(7) || 8080);
const isolate = args.includes("--isolate");
// --cors: answer like a CDN (Access-Control-Allow-Origin: *), to test cross-origin embedding
const cors = args.includes("--cors");
const types = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm", ".svg": "image/svg+xml",
  ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp", ".woff2": "font/woff2", ".kevala": "application/octet-stream",
  ".md": "text/markdown; charset=utf-8",
};

createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  let path = normalize(join(root, decodeURIComponent(url.pathname)));
  if (!path.startsWith(root)) return res.writeHead(403).end();
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

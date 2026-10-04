// Maps a request path to a file under the served directory.
import { join, normalize, sep } from "node:path";

/**
 * The file `pathname` (the URL path, still percent-encoded) names under `root`, or an HTTP status:
 * 400 for a path that does not decode, 403 for one that leaves `root`.
 */
export function resolveRequestPath(root, pathname) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return { status: 400 };
  }
  if (decoded.includes("\0")) return { status: 400 };
  const path = normalize(join(root, decoded));
  // A sibling whose name starts with the root's name (root "/srv/site", path "/srv/site-private")
  // shares the prefix, so compare whole path segments.
  if (path !== root && !path.startsWith(root.endsWith(sep) ? root : root + sep)) return { status: 403 };
  return { path };
}

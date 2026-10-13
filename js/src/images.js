import { checkImageDimensions, encodedImageInfo } from "./image-header.js";

const MAX_BYTES = 32 * 1024 * 1024;

/** Decode browser image parts into RGBA pixels without serializing pixels as JSON. */
export async function imagePixels(data, { signal } = {}) {
  signal?.throwIfAborted();
  if (data && Number.isInteger(data.width) && Number.isInteger(data.height) && data.data != null) {
    const { width, height } = data;
    checkImageDimensions(width, height);
    if (!(data.data instanceof Uint8Array || data.data instanceof Uint8ClampedArray) || data.data.length !== width * height * 4) throw new Error("image data must contain width × height RGBA bytes");
    return { width, height, data: data.data.slice() };
  }
  let blob;
  if (data instanceof Blob) blob = data;
  else if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) blob = new Blob([data]);
  else if (typeof data === "string" || data instanceof URL) {
    const url = new URL(String(data), globalThis.location?.href);
    if (!["http:", "https:", "blob:", "data:"].includes(url.protocol)) throw new Error("image URL must use HTTP, HTTPS, blob, or data");
    const response = await fetch(url, { signal });
    if (!response.ok) throw new Error(`image download failed: HTTP ${response.status}`);
    const length = Number(response.headers.get("content-length"));
    if (length > MAX_BYTES) { await response.body?.cancel(); throw new Error("image download exceeds 32 MiB"); }
    const reader = response.body?.getReader();
    if (!reader) blob = await response.blob();
    else {
      const chunks = [];
      let size = 0;
      try {
        for (;;) {
          signal?.throwIfAborted();
          const { done, value } = await reader.read();
          if (done) break;
          size += value.byteLength;
          if (size > MAX_BYTES) throw new Error("image download exceeds 32 MiB");
          chunks.push(value);
        }
      } finally { await reader.cancel(); }
      blob = new Blob(chunks);
    }
  } else throw new TypeError("image parts need ImageData, RGBA pixels, a Blob, encoded bytes, or a URL");
  if (blob.size > MAX_BYTES) throw new Error("encoded image exceeds 32 MiB");
  const info = encodedImageInfo(new Uint8Array(await blob.arrayBuffer()));
  signal?.throwIfAborted();
  // Browser bitmap and WebCodecs paths can discard RGB at alpha=0 and round partial-alpha RGB.
  // Raw RGBA stays exact; refusing encoded alpha is preferable to changing the model's pixels.
  if (info.alpha) throw new Error("alpha-bearing encoded images are unsupported because browser decoding can change RGB; pass ImageData or raw RGBA to preserve RGB while ignoring alpha");
  if (typeof createImageBitmap !== "function" || typeof OffscreenCanvas !== "function") throw new Error("this browser cannot decode image files; pass ImageData or RGBA pixels instead");
  const bitmap = await createImageBitmap(new Blob([blob], { type: info.type }), { colorSpaceConversion: "none" });
  try {
    signal?.throwIfAborted();
    checkImageDimensions(bitmap.width, bitmap.height);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("image decoding requires a 2D canvas");
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height);
    context.clearRect(0, 0, bitmap.width, bitmap.height);
    for (let i = 3; i < pixels.data.length; i += 4) {
      if (pixels.data[i] !== 255) throw new Error("decoded image contains transparency; pass ImageData or raw RGBA to preserve RGB while ignoring alpha");
    }
    return pixels;
  } finally { bitmap.close(); }
}

/** The image budget is per request, including a batched request. */
export function assertImageCount(request) {
  const count = request.parts?.filter((part) => part?.type === "image").length || 0;
  if (count > 4) throw new Error("a request supports at most four images");
  return count;
}

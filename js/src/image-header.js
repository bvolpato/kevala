// Inspect encoded dimensions before asking a browser to allocate decoded pixels.
// PNG IHDR/tRNS: https://www.w3.org/TR/png-3/
// WebP VP8/VP8L/VP8X: https://developers.google.com/speed/webp/docs/riff_container

export function checkImageDimensions(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 8192 || height > 8192 || width * height > 16 * 1024 * 1024) throw new Error("image dimensions exceed 8192px per side or the 16 megapixel limit");
}

const invalid = () => new Error("invalid or truncated encoded image header");
const ascii = (b, at, n) => String.fromCharCode(...b.subarray(at, at + n));
const u24 = (b, at) => b[at] + b[at + 1] * 256 + b[at + 2] * 65536;

/** Only raster formats whose allocation dimensions and alpha can be checked are decoded. */
export function encodedImageInfo(b) {
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const size = (width, height) => { checkImageDimensions(width, height); return { width, height }; };
  if (b.length >= 8 && b[0] === 137 && ascii(b, 1, 7) === "PNG\r\n\x1a\n") {
    let info;
    let alpha = false;
    let imageData = false;
    for (let at = 8; at < b.length;) {
      if (at + 12 > b.length) throw invalid();
      const n = v.getUint32(at), end = at + 12 + n;
      if (end > b.length) throw invalid();
      const chunk = ascii(b, at + 4, 4);
      if (chunk === "IHDR") {
        if (at !== 8 || n !== 13) throw invalid();
        info = size(v.getUint32(at + 8), v.getUint32(at + 12));
        const color = b[at + 17];
        if (![0, 2, 3, 4, 6].includes(color)) throw invalid();
        alpha = color === 4 || color === 6;
      } else if (!info) throw invalid();
      if (chunk === "tRNS") alpha = true;
      if (chunk === "fcTL") {
        if (n !== 26) throw invalid();
        const frame = size(v.getUint32(at + 12), v.getUint32(at + 16));
        if (frame.width > info.width || frame.height > info.height) throw invalid();
      }
      if (chunk === "IDAT") imageData = true;
      if (chunk === "IEND") {
        if (n || !imageData) throw invalid();
        return { ...info, alpha, type: "image/png" };
      }
      at = end;
    }
    throw invalid();
  }
  if (b.length >= 2 && b[0] === 255 && b[1] === 216) {
    let info;
    for (let at = 2; at < b.length;) {
      if (b[at++] !== 255) throw invalid();
      while (b[at] === 255) at++;
      const marker = b[at++];
      if (marker === 217) throw invalid();
      if (marker === 218) {
        if (!info || at + 2 > b.length) throw invalid();
        const n = v.getUint16(at);
        if (n < 6 || at + n > b.length) throw invalid();
        return { ...info, alpha: false, type: "image/jpeg" };
      }
      if (marker === 1 || marker >= 208 && marker <= 215) continue;
      if (at + 2 > b.length) throw invalid();
      const n = v.getUint16(at);
      if (n < 2 || at + n > b.length) throw invalid();
      if (marker >= 192 && marker <= 207 && ![196, 200, 204].includes(marker)) {
        if (n < 8) throw invalid();
        info = size(v.getUint16(at + 5), v.getUint16(at + 3));
      }
      at += n;
    }
    throw invalid();
  }
  if (b.length >= 12 && ascii(b, 0, 4) === "RIFF" && ascii(b, 8, 4) === "WEBP") {
    const fileEnd = v.getUint32(4, true) + 8;
    if (fileEnd !== b.length) throw invalid();
    let info;
    let alpha = false;
    let imageData = false;
    const chunks = (start, end, frame = false) => {
      for (let at = start; at < end;) {
        if (at + 8 > end) throw invalid();
        const chunk = ascii(b, at, 4), n = v.getUint32(at + 4, true), data = at + 8;
        const next = data + n + (n & 1);
        if (next > end) throw invalid();
        let dimensions;
        if (chunk === "VP8X") {
          if (frame || at !== 12 || n !== 10) throw invalid();
          info = size(u24(b, data + 4) + 1, u24(b, data + 7) + 1);
          alpha ||= !!(b[data] & 16);
        } else if (chunk === "VP8L") {
          if (n < 5 || b[data] !== 47) throw invalid();
          const bits = v.getUint32(data + 1, true);
          if (bits >>> 29) throw invalid();
          dimensions = size((bits & 16383) + 1, ((bits >>> 14) & 16383) + 1);
          // A clear alpha hint is verified by the decoded-pixel scan before pixels are returned.
          alpha ||= !!(bits & 0x10000000);
        } else if (chunk === "VP8 ") {
          if (n < 10 || ascii(b, data + 3, 3) !== "\x9d\x01\x2a") throw invalid();
          dimensions = size(v.getUint16(data + 6, true) & 16383, v.getUint16(data + 8, true) & 16383);
        } else if (chunk === "ALPH") alpha = true;
        else if (chunk === "ANIM") {
          if (n !== 6) throw invalid();
          alpha ||= b[data + 3] !== 255;
        }
        else if (chunk === "ANMF") {
          if (frame || !info || n < 16) throw invalid();
          const dimensions = size(u24(b, data + 6) + 1, u24(b, data + 9) + 1);
          if (u24(b, data) * 2 + dimensions.width > info.width || u24(b, data + 3) * 2 + dimensions.height > info.height) throw invalid();
          chunks(data + 16, data + n, true);
        }
        if (dimensions) {
          if (info && (dimensions.width > info.width || dimensions.height > info.height)) throw invalid();
          if (!frame) info ||= dimensions;
          imageData = true;
        }
        at = next;
      }
    };
    chunks(12, fileEnd);
    if (!info || !imageData) throw invalid();
    return { ...info, alpha, type: "image/webp" };
  }
  throw new Error("encoded images must use PNG, JPEG, or WebP; pass ImageData or raw RGBA for other formats");
}

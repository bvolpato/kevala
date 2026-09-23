/** Use the same runtime as the page, including models not yet in the npm release. */
export const RUNTIME_URL = new URL("../js/src/index.js", import.meta.url).href;

/** Format JSON values as JavaScript without changing property names or string contents. */
export function js(value, indent = "") {
  const inner = indent + "  ";
  const fitsOneLine = (s) => s.length < 80 && !s.includes("\n");
  if (Array.isArray(value)) {
    const items = value.map((x) => js(x, inner));
    const oneLine = `[${items.join(", ")}]`;
    return fitsOneLine(oneLine) ? oneLine : `[\n${items.map((x) => inner + x).join(",\n")},\n${indent}]`;
  }
  if (value && typeof value === "object") {
    const key = (k) => k === "__proto__" ? '["__proto__"]' : /^[A-Za-z_$][\w$]*$/.test(k) ? k : JSON.stringify(k);
    const entries = Object.entries(value).map(([k, x]) => `${key(k)}: ${js(x, inner)}`);
    const oneLine = `{ ${entries.join(", ")} }`;
    return fitsOneLine(oneLine) ? oneLine : `{\n${entries.map((x) => inner + x).join(",\n")},\n${indent}}`;
  }
  return JSON.stringify(value);
}

export function loadCode(selection) {
  const model = selection.model === "custom" ? selection.customUrl || "https://example.com/model.kevala" : selection.model;
  const options = { model };
  if (selection.backend && selection.backend !== "auto") options.backend = selection.backend;
  if (selection.from && selection.from !== "pack") options.from = selection.from;
  return `import { Kevala } from ${JSON.stringify(RUNTIME_URL)};\n\nconst kevala = await Kevala.load(${js(options)});`;
}

export function requestCode(request, selection) {
  const setup = `${loadCode(selection)}\n\nconst questions = ${js(request.questions)};\n\n`;
  if (request.items.length === 1) {
    return `${setup}const state = ${js(request.items[0].state)};\n\n` +
      "const result = await kevala.decide(state, questions);\nconsole.log(result.answers);";
  }
  return `${setup}const states = ${js(request.items.map((item) => item.state))};\n\n` +
    "const results = await kevala.decideMany(states.map((state) => ({ state, questions })));\n" +
    "results.forEach((result) => console.log(result.answers));";
}

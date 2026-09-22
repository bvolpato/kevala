// A plain textarea with a syntax-highlighted twin behind it: the textarea's text is
// transparent, so the caret and selection are native while the colors come from the <pre>.
// JSON is highlighted as you type; plain text stays plain. It grows with its content.

import { highlightJSON, esc } from "./ui.js";

export class Editor {
  #options;
  #textarea;
  #highlighted;

  /**
   * @param {HTMLElement} el container
   * @param {{
   *   value?: string,
   *   mode?: "json" | "auto",
   *   label?: string,
   *   minRows?: number,
   *   onInput?: (v: string) => void,
   *   onRun?: () => void,
   * }} options
   */
  constructor(el, options = {}) {
    this.#options = options;
    this.el = el;
    el.classList.add("editor");
    const label = esc(options.label || "Editor");
    el.innerHTML =
      `<pre class="ed-hl" aria-hidden="true"></pre>` +
      `<textarea class="ed-ta" spellcheck="false" autocapitalize="off" autocomplete="off" aria-label="${label}"></textarea>`;
    this.#highlighted = el.querySelector(".ed-hl");
    this.#textarea = el.querySelector(".ed-ta");
    const ta = this.#textarea;
    ta.style.minHeight = `${(options.minRows || 3) * 1.6 + 1.6}em`;
    ta.addEventListener("input", () => {
      this.paint();
      options.onInput?.(ta.value);
    });
    ta.addEventListener("scroll", () => {
      this.#highlighted.scrollTop = ta.scrollTop;
      this.#highlighted.scrollLeft = ta.scrollLeft;
    });
    ta.addEventListener("keydown", (e) => this.#onKeyDown(e));
    this.value = options.value || "";
  }

  #onKeyDown(e) {
    const ta = this.#textarea;
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      this.#options.onRun?.();
    } else if (e.key === "Tab" && !e.shiftKey) {
      e.preventDefault();
      this.#insert("  ");
    } else if (e.key === "Enter" && !e.shiftKey && !e.altKey) {
      // keep the indentation of the current line, one level deeper after an opening bracket
      const before = ta.value.slice(0, ta.selectionStart);
      const line = before.slice(before.lastIndexOf("\n") + 1);
      const indent = line.match(/^\s*/)[0] + (/[{[]\s*$/.test(line) ? "  " : "");
      if (!indent) return;
      e.preventDefault();
      this.#insert("\n" + indent);
    }
  }

  /** Replaces the selection with `text`, as typing it would. */
  #insert(text) {
    const ta = this.#textarea;
    ta.setRangeText(text, ta.selectionStart, ta.selectionEnd, "end");
    this.paint();
    this.#options.onInput?.(ta.value);
  }

  get value() {
    return this.#textarea.value;
  }

  set value(v) {
    this.#textarea.value = v;
    this.paint();
  }

  /** Whether the text reads as JSON (an object, array, number, string literal...). */
  isJSON() {
    const text = this.#textarea.value.trim();
    return this.#options.mode === "json" || /^[{["]/.test(text);
  }

  paint() {
    const ta = this.#textarea;
    const v = ta.value;
    // a trailing newline needs a character after it to take up height in the <pre>
    this.#highlighted.innerHTML = (this.isJSON() ? highlightJSON(v) : esc(v)) + "\n ";
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight + 2}px`;
  }

  setInvalid(msg) {
    this.el.classList.toggle("invalid", !!msg);
    this.el.title = msg || "";
  }

  focus() {
    this.#textarea.focus();
  }
}

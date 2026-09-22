// A plain textarea with a syntax-highlighted twin behind it: the textarea's text is
// transparent, so the caret and selection are native while the colors come from the <pre>.
// JSON is highlighted as you type; plain text stays plain. It grows with its content.

import { highlightJSON, esc } from "./ui.js";

export class Editor {
  /**
   * @param {HTMLElement} el container
   * @param {{ value?: string, mode?: "json" | "auto", label?: string, minRows?: number, onInput?: (v: string) => void, onRun?: () => void }} o
   */
  constructor(el, o = {}) {
    this.o = o;
    this.el = el;
    el.classList.add("editor");
    el.innerHTML = `<pre class="ed-hl" aria-hidden="true"></pre><textarea class="ed-ta" spellcheck="false" autocapitalize="off" autocomplete="off" aria-label="${esc(o.label || "Editor")}"></textarea>`;
    this.hl = el.querySelector(".ed-hl");
    this.ta = el.querySelector(".ed-ta");
    this.ta.style.minHeight = `${(o.minRows || 3) * 1.6 + 1.6}em`;
    this.ta.addEventListener("input", () => {
      this.paint();
      o.onInput?.(this.ta.value);
    });
    this.ta.addEventListener("scroll", () => {
      this.hl.scrollTop = this.ta.scrollTop;
      this.hl.scrollLeft = this.ta.scrollLeft;
    });
    this.ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        o.onRun?.();
      } else if (e.key === "Tab" && !e.shiftKey) {
        e.preventDefault();
        this.ta.setRangeText("  ", this.ta.selectionStart, this.ta.selectionEnd, "end");
        this.paint();
        o.onInput?.(this.ta.value);
      } else if (e.key === "Enter" && !e.shiftKey && !e.altKey) {
        // keep the indentation of the current line
        const v = this.ta.value;
        const s = this.ta.selectionStart;
        const line = v.slice(v.lastIndexOf("\n", s - 1) + 1, s);
        const indent = line.match(/^\s*/)[0] + (/[{[]\s*$/.test(line) ? "  " : "");
        if (!indent) return;
        e.preventDefault();
        this.ta.setRangeText("\n" + indent, s, this.ta.selectionEnd, "end");
        this.paint();
        o.onInput?.(this.ta.value);
      }
    });
    this.value = o.value || "";
  }

  get value() {
    return this.ta.value;
  }

  set value(v) {
    this.ta.value = v;
    this.paint();
  }

  /** Whether the text reads as JSON (an object, array, number, string literal...). */
  isJSON() {
    const t = this.ta.value.trim();
    return this.o.mode === "json" || /^[{["]/.test(t);
  }

  paint() {
    const v = this.ta.value;
    // a trailing newline needs a character after it to take up height in the <pre>
    this.hl.innerHTML = (this.isJSON() ? highlightJSON(v) : esc(v)) + "\n ";
    this.ta.style.height = "auto";
    this.ta.style.height = `${this.ta.scrollHeight + 2}px`;
  }

  setInvalid(msg) {
    this.el.classList.toggle("invalid", !!msg);
    this.el.title = msg || "";
  }

  focus() {
    this.ta.focus();
  }
}

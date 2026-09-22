// Inbox triage: twelve emails and the laya SDK's email questions, sent in growing batches so
// rows fill in as answers arrive. Sort, filter, open one to see every probability.

import { renderAnswers, fmtMs, esc, highlight, wireCopy, modelGate, backendLabel, css, decideStream, CDN, js } from "../ui.js";

// the laya SDK's email questions, written out
const Q = {
  category: {
    type: "choice",
    instructions: "Which team should handle the email in `body`?",
    criteria: {
      billing: "invoices, payments, refunds",
      technical: "bugs, outages, integrations",
      sales: "pricing, demos, new purchases",
      security: "phishing, scams, account compromise",
      hr: "hiring, leave, payroll",
      other: "none of the above",
    },
  },
  is_spam: {
    type: "noul",
    instructions: "Is this email unsolicited spam or bulk marketing?",
  },
  is_phishing: {
    type: "noul",
    instructions: "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
    criteria: { true: "phishing, scam, or fraud", false: "a legitimate email" },
  },
  urgency: {
    type: "score",
    instructions: "How urgent is the request in `body`?",
    criteria: ["no time pressure", "needs attention soon", "blocking issue or hard deadline"],
  },
  needs_reply: { type: "noul", instructions: "Does the sender expect a reply?" },
};

const now = Date.now();
const ago = (m) => now - m * 60000;
const SAMPLE = [
  { from: "Priya Natarajan <priya@northwind-logistics.com>", subject: "Checkout is down for all EU customers", body: "Since 09:10 CET every checkout in the EU region fails with a 502 after payment authorization. We are losing orders every minute. Can someone from your platform team join our bridge right now? Incident link is in the ticket.", at: ago(4) },
  { from: "Billing <billing@acme-saas.io>", subject: "Invoice INV-20431 for September", body: "Hi, attached is your invoice for September (USD 1,240.00), due on October 15. No action is needed if autopay is enabled. Reply to this email if you have questions about the charges.", at: ago(38) },
  { from: "IT Security <security-alerts@micros0ft-support.co>", subject: "Unusual sign-in: verify within 2 hours", body: "We detected an unusual sign-in to your mailbox. Your account will be locked in 2 hours unless you confirm your password at the secure portal: http://micros0ft-support.co/verify. Do not ignore this message.", at: ago(12) },
  { from: "Marcus Lee <marcus.lee@brightpath.edu>", subject: "Pricing for 400 seats and a demo next week?", body: "Hello, our district is evaluating tools for the spring term. Could you send pricing for roughly 400 teacher seats and set up a 30 minute demo next Tuesday or Wednesday afternoon?", at: ago(95) },
  { from: "Jenna Park <jenna.park@ourcompany.com>", subject: "Parental leave dates", body: "Hi HR team, my due date moved up. I'd like to start parental leave on November 3 instead of November 17. What do I need to update in the portal, and does this change my payroll schedule?", at: ago(160) },
  { from: "Deals Daily <offers@mega-deals-now.biz>", subject: "🔥 72 hours only: 80% off smart watches", body: "Exclusive flash sale for our valued subscribers! Grab premium smart watches at 80% off. Limited stock. Click to shop now. Unsubscribe anytime.", at: ago(210) },
  { from: "Tom Becker <tom@beckerandsons.de>", subject: "Charged twice this month", body: "Hi, I was charged twice for the Pro plan on September 3 (two identical charges of 49 EUR). Please refund one of them. Thanks, Tom", at: ago(55) },
  { from: "GitHub <noreply@github.com>", subject: "[api-gateway] Build failed on main", body: "The workflow 'deploy' failed on main for commit 8c1f2e0: step 'integration-tests' exited with code 1. Releases are blocked until main is green.", at: ago(22) },
  { from: "CEO Office <ceo.office.private@gmail.com>", subject: "Quick favor, confidential", body: "Are you at your desk? I need you to buy six 200 USD gift cards for a client event today and send me the codes by email. Keep this between us, I'm in meetings all day.", at: ago(8) },
  { from: "Lena Hoffmann <lena@hoffmann-design.studio>", subject: "Webhook retries after our API key rotation", body: "After rotating our API key yesterday, webhook deliveries to our endpoint return 401 and your dashboard shows them retrying. Did the signing secret change as well? Not urgent, but we'd like to fix it this week.", at: ago(300) },
  { from: "Community Team <newsletter@devconf.events>", subject: "DevConf 2026: speaker lineup is live", body: "The full speaker lineup for DevConf 2026 is now online, with early-bird tickets available until the end of the month. We hope to see you in Lisbon!", at: ago(420) },
  { from: "Rafael Souza <rafael.souza@ourcompany.com>", subject: "Contract renewal needs signature by Friday", body: "The renewal with Globex expires Friday at midnight. Legal approved the redlines; it just needs your signature in the e-sign tool. If it lapses we lose the discounted rate.", at: ago(70) },
];

const COLORS = ["#7aa2ff", "#45e0c0", "#c592ff", "#f6c453", "#ff9f5a", "#8be36b", "#ff7a8a"];
const initials = (from) => from.replace(/<.*>/, "").trim().split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();
const name = (from) => from.replace(/<.*>/, "").trim() || from;
const when = (t) => {
  const m = Math.max(0, Math.round((Date.now() - t) / 60000));
  return m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
};
const hash = (s) => [...s].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7);

// expected urgency on the 0..2 scale, from the score answer
function urgencyOf(a) {
  return typeof a.score === "number" ? a.score : Object.entries(a.probabilities).reduce((s, [k, p]) => s + Number(k) * p, 0);
}

const CODE = `import { Kevala } from "${CDN}";

const kevala = await Kevala.load({ model: "laya" });

const questions = ${js(Q)};

const emails = [
  { from: "billing@vendor.example", subject: "Invoice overdue", body: "Your invoice INV-2231 is 30 days overdue. Please pay by Friday." },
  { from: "security@micros0ft-support.co", subject: "Verify within 2 hours", body: "Confirm your password at http://micros0ft-support.co/verify or lose access." },
];

// every email is its own state; one call runs them together
const results = await kevala.decideMany(emails.map((email) => ({ state: email, questions })));
results.forEach((r, i) => {
  const a = r.answers;
  console.log(emails[i].subject, a.category.choice, "urgency", a.urgency.score, "P(phishing)", a.is_phishing.noul);
});`;

const TEMPLATE = `<div class="wrap">
  <div class="page-head">
    <div class="eyebrow">Demo · batching</div>
    <h1>Inbox triage</h1>
    <p>Twelve emails, the laya SDK's <code>email</code> questions (team, spam, phishing, urgency, needs a reply), batched with <code>decideMany</code>. Rows fill in as each batch returns, so the first answers show after one short pass. Sort by urgency, open one to see every probability.</p>
  </div>

  <div data-f="gate"></div>

  <div class="toolbar">
    <button type="button" class="btn primary" data-f="run" disabled>Triage inbox</button>
    <div class="seg" role="group" aria-label="Sort">
      <button type="button" data-sort="urgency" aria-pressed="true">Most urgent</button>
      <button type="button" data-sort="phishing" aria-pressed="false">Riskiest</button>
      <button type="button" data-sort="time" aria-pressed="false">Newest</button>
    </div>
    <select data-f="filter" aria-label="Team filter"><option value="">All teams</option>${Object.keys(Q.category.criteria)
      .map((k) => `<option value="${esc(k)}">${esc(k)}</option>`)
      .join("")}</select>
    <span class="spacer"></span>
    <span class="run-meta" data-f="meta"></span>
  </div>

  <ol class="mail" data-f="mail"></ol>

  <section class="tight">
    <div class="grid-2">
      <details class="card pad add-card">
        <summary><b>Add your own email</b><span class="tiny faint">it is triaged with the others</span></summary>
        <div class="add">
          <input type="text" data-f="a-from" aria-label="From" placeholder="From, e.g. Ana Ruiz <ana@example.com>">
          <input type="text" data-f="a-subj" aria-label="Subject" placeholder="Subject">
          <textarea data-f="a-body" rows="4" aria-label="Body" placeholder="Body"></textarea>
          <div><button type="button" class="btn" data-f="a-add">Add and re-triage</button></div>
        </div>
      </details>
      <div class="card pad">
        <h3>The call</h3>
        <div class="code"><pre data-f="code"></pre></div>
      </div>
    </div>
  </section>
  <p class="tiny faint view-foot">Sample emails are fictional. Questions: the laya SDK's email set, written out in the code above.</p>
</div>`;

export function mount(el, { session }) {
  css(new URL("./inbox.css", import.meta.url).href);
  el.innerHTML = TEMPLATE;
  const $ = (f) => el.querySelector(`[data-f="${f}"]`);
  const mail = $("mail");
  const runBtn = $("run");

  modelGate($("gate"), "triage the inbox");
  $("code").innerHTML = highlight(CODE);
  wireCopy(el);

  let emails = SAMPLE.slice();
  let results = null;
  let streaming = false;
  let sort = "urgency";
  let filter = "";
  const open = new Set();

  function render() {
    const items = emails.map((e, i) => ({ e, i, r: results?.[i] }));
    // while answers stream in, rows keep their place; they sort once every email is scored
    const key = streaming ? (x) => -x.e.at : {
      urgency: (x) => (x.r ? -urgencyOf(x.r.answers.urgency) : x.e.at * -1e-12),
      phishing: (x) => (x.r ? -Math.max(x.r.answers.is_phishing.noul, x.r.answers.is_spam.noul * 0.5) : 0),
      time: (x) => -x.e.at,
    }[sort];
    items.sort((a, b) => key(a) - key(b) || b.e.at - a.e.at);
    const shown = items.filter((x) => !filter || x.r?.answers.category.choice === filter);
    mail.innerHTML = shown
      .map(({ e, i, r }) => {
        const a = r?.answers;
        const u = a ? urgencyOf(a.urgency) : 0;
        const legend = a ? a.urgency.legend?.[Math.round(u)] ?? "" : "";
        const tags = a
          ? [
              a.is_phishing.noul >= 0.5 ? `<span class="tag phish" title="P(phishing) ${a.is_phishing.noul.toFixed(3)}">phishing ${(a.is_phishing.noul * 100).toFixed(0)}%</span>` : "",
              a.is_spam.noul >= 0.5 ? `<span class="tag spam" title="P(spam) ${a.is_spam.noul.toFixed(3)}">spam ${(a.is_spam.noul * 100).toFixed(0)}%</span>` : "",
              a.needs_reply.noul >= 0.5 ? `<span class="tag reply" title="P(needs reply) ${a.needs_reply.noul.toFixed(3)}">reply ${(a.needs_reply.noul * 100).toFixed(0)}%</span>` : "",
            ].join("")
          : "";
        const cat = a?.category;
        return `<li class="m ${a ? "" : "pending"}" data-i="${i}">
        <button type="button" aria-expanded="${open.has(i)}">
          <span class="av" style="background:${COLORS[hash(e.from) % COLORS.length]}">${esc(initials(e.from))}</span>
          <span style="min-width:0"><span class="who"><b>${esc(name(e.from))}</b><span class="time">${esc(when(e.at))}</span></span><span class="subj">${esc(e.subject)}</span><span class="snip">${esc(e.body)}</span><span class="tags">${tags}</span></span>
          <span class="urg"><span class="ul"><span>${a ? esc(legend) : "urgency"}</span><span>${a ? u.toFixed(2) + " / 2" : "–"}</span></span><span class="track"><i data-w="${((u / 2) * 100).toFixed(1)}"></i></span></span>
          <span class="team">${cat ? `${esc(cat.choice)}<span>${(cat.probabilities[cat.choice] * 100).toFixed(0)}%</span>` : `<span>team</span>`}</span>
        </button>
        ${open.has(i) ? `<div class="detail"><div><div class="tiny faint">${esc(e.from)}</div><div class="body">${esc(e.body)}</div></div><div class="answers" data-ans="${i}"></div></div>` : ""}
      </li>`;
      })
      .join("");
    if (!shown.length) mail.innerHTML = `<li class="none muted small">${results ? `No email is routed to ${esc(filter)}.` : "Triage the inbox to filter by team."}</li>`;
    requestAnimationFrame(() => mail.querySelectorAll(".urg .track i").forEach((t) => (t.style.width = `${t.dataset.w}%`)));
    for (const i of open) {
      const box = mail.querySelector(`[data-ans="${i}"]`);
      if (box && results?.[i]) renderAnswers(box, results[i], Q);
    }
  }

  mail.addEventListener("click", (e) => {
    const li = e.target.closest(".m");
    if (!li || e.target.closest(".detail")) return;
    const i = Number(li.dataset.i);
    open.has(i) ? open.delete(i) : open.add(i);
    render();
  });
  for (const b of el.querySelectorAll("[data-sort]")) {
    b.addEventListener("click", () => {
      sort = b.dataset.sort;
      for (const x of el.querySelectorAll("[data-sort]")) x.setAttribute("aria-pressed", String(x === b));
      render();
    });
  }
  $("filter").addEventListener("change", (e) => {
    filter = e.target.value;
    render();
  });

  // -------------------------------------------------------------------------------------------
  // one triage in flight; `owed` means the results are behind the list, so a triage that is due
  // while the view is hidden waits for show()

  let kevala = null;
  let visible = false;
  let running = false;
  let owed = false;
  let lastRun = null;

  function request() {
    owed = true;
    triage();
  }

  async function triage() {
    if (!kevala || !visible || running || !owed) return;
    owed = false;
    running = true;
    const w = kevala;
    const list = emails;
    runBtn.disabled = true;
    $("meta").textContent = `Triaging ${list.length} emails…`;
    results = null;
    render();
    try {
      const t0 = performance.now();
      // every email is a separate state with the same questions, sent in growing batches
      // (1, 2, 4, ...) so the first rows fill in after one short pass
      let first = 0;
      let passes = 0;
      streaming = true;
      results = new Array(list.length);
      const stale = () => w !== kevala || list !== emails;
      let done = 0;
      const res = await decideStream(
        w,
        list.map((e) => ({ state: { from: e.from, subject: e.subject, body: e.body }, questions: Q })),
        (r, i) => {
          results[i] = r;
          if (!first) first = performance.now() - t0;
          if (++done === 1 || (done & (done - 1)) === 0 || done === list.length) {
            passes++;
            $("meta").textContent = `Triaging… ${done} of ${list.length}`;
            render();
          }
        },
        { isStale: stale },
      );
      streaming = false;
      const ms = performance.now() - t0;
      if (res && !stale()) {
        results = res;
        const tokens = res.reduce((s, r) => s + (r.usage?.input_tokens || 0), 0);
        $("meta").innerHTML = `<b>${list.length}</b> emails · first answer in <b>${fmtMs(first)}</b> · all in <b>${fmtMs(ms)}</b> (${passes} passes) · ${tokens.toLocaleString()} tokens · ${esc(backendLabel(w.info))}`;
        lastRun = { ms, first, tokens, n: list.length, backend: w.info.backend };
      }
    } catch (e) {
      streaming = false;
      if (w === kevala) $("meta").textContent = `Error: ${e.message}`;
    } finally {
      running = false;
      runBtn.disabled = !kevala;
      render();
      triage();
    }
  }

  runBtn.addEventListener("click", request);
  $("a-add").addEventListener("click", () => {
    const body = $("a-body").value.trim();
    if (!body) return;
    emails = [{ from: $("a-from").value.trim() || "You <you@example.com>", subject: $("a-subj").value.trim() || "(no subject)", body, at: Date.now() }, ...emails];
    // indexes shift by one: keep the open emails open
    const was = [...open];
    open.clear();
    for (const i of was) open.add(i + 1);
    $("a-body").value = "";
    results = null;
    if (kevala) request();
    else render();
  });

  const reset = () => {
    owed = false;
    results = null;
    lastRun = null;
    runBtn.disabled = !kevala || running;
    $("meta").textContent = kevala ? "" : "Load a model, then triage.";
    render();
  };
  const sync = (s) => {
    const w = s.ready ? s.kevala : null;
    if (w === kevala) return;
    kevala = w;
    reset();
    if (w) request();
  };
  reset();
  session.on(sync);
  sync(session);

  return {
    show() {
      visible = true;
      triage();
    },
    hide() {
      visible = false;
    },
    // read-only handles for tests
    get last() {
      return lastRun;
    },
    get results() {
      return results;
    },
  };
}

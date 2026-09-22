// Inbox triage: twelve emails and the laya SDK's email questions, sent in growing batches so
// rows fill in as answers arrive. Sort, filter, open one to see every probability.

import { renderAnswers, fmtMs, esc, highlight, wireCopy, modelGate, backendLabel, css, decideStream, CDN, js } from "../ui.js";

// the laya SDK's email questions, written out
const QUESTIONS = {
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
const minutesAgo = (m) => now - m * 60000;
const SAMPLE = [
  {
    from: "Priya Natarajan <priya@northwind-logistics.com>",
    subject: "Checkout is down for all EU customers",
    body: "Since 09:10 CET every checkout in the EU region fails with a 502 after payment authorization. We are losing orders every minute. Can someone from your platform team join our bridge right now? Incident link is in the ticket.",
    at: minutesAgo(4),
  },
  {
    from: "Billing <billing@acme-saas.io>",
    subject: "Invoice INV-20431 for September",
    body: "Hi, attached is your invoice for September (USD 1,240.00), due on October 15. No action is needed if autopay is enabled. Reply to this email if you have questions about the charges.",
    at: minutesAgo(38),
  },
  {
    from: "IT Security <security-alerts@micros0ft-support.co>",
    subject: "Unusual sign-in: verify within 2 hours",
    body: "We detected an unusual sign-in to your mailbox. Your account will be locked in 2 hours unless you confirm your password at the secure portal: http://micros0ft-support.co/verify. Do not ignore this message.",
    at: minutesAgo(12),
  },
  {
    from: "Marcus Lee <marcus.lee@brightpath.edu>",
    subject: "Pricing for 400 seats and a demo next week?",
    body: "Hello, our district is evaluating tools for the spring term. Could you send pricing for roughly 400 teacher seats and set up a 30 minute demo next Tuesday or Wednesday afternoon?",
    at: minutesAgo(95),
  },
  {
    from: "Jenna Park <jenna.park@ourcompany.com>",
    subject: "Parental leave dates",
    body: "Hi HR team, my due date moved up. I'd like to start parental leave on November 3 instead of November 17. What do I need to update in the portal, and does this change my payroll schedule?",
    at: minutesAgo(160),
  },
  {
    from: "Deals Daily <offers@mega-deals-now.biz>",
    subject: "🔥 72 hours only: 80% off smart watches",
    body: "Exclusive flash sale for our valued subscribers! Grab premium smart watches at 80% off. Limited stock. Click to shop now. Unsubscribe anytime.",
    at: minutesAgo(210),
  },
  {
    from: "Tom Becker <tom@beckerandsons.de>",
    subject: "Charged twice this month",
    body: "Hi, I was charged twice for the Pro plan on September 3 (two identical charges of 49 EUR). Please refund one of them. Thanks, Tom",
    at: minutesAgo(55),
  },
  {
    from: "GitHub <noreply@github.com>",
    subject: "[api-gateway] Build failed on main",
    body: "The workflow 'deploy' failed on main for commit 8c1f2e0: step 'integration-tests' exited with code 1. Releases are blocked until main is green.",
    at: minutesAgo(22),
  },
  {
    from: "CEO Office <ceo.office.private@gmail.com>",
    subject: "Quick favor, confidential",
    body: "Are you at your desk? I need you to buy six 200 USD gift cards for a client event today and send me the codes by email. Keep this between us, I'm in meetings all day.",
    at: minutesAgo(8),
  },
  {
    from: "Lena Hoffmann <lena@hoffmann-design.studio>",
    subject: "Webhook retries after our API key rotation",
    body: "After rotating our API key yesterday, webhook deliveries to our endpoint return 401 and your dashboard shows them retrying. Did the signing secret change as well? Not urgent, but we'd like to fix it this week.",
    at: minutesAgo(300),
  },
  {
    from: "Community Team <newsletter@devconf.events>",
    subject: "DevConf 2026: speaker lineup is live",
    body: "The full speaker lineup for DevConf 2026 is now online, with early-bird tickets available until the end of the month. We hope to see you in Lisbon!",
    at: minutesAgo(420),
  },
  {
    from: "Rafael Souza <rafael.souza@ourcompany.com>",
    subject: "Contract renewal needs signature by Friday",
    body: "The renewal with Globex expires Friday at midnight. Legal approved the redlines; it just needs your signature in the e-sign tool. If it lapses we lose the discounted rate.",
    at: minutesAgo(70),
  },
];

const AVATAR_COLORS = ["#7aa2ff", "#45e0c0", "#c592ff", "#f6c453", "#ff9f5a", "#8be36b", "#ff7a8a"];
const senderName = (from) => from.replace(/<.*>/, "").trim();
const displayName = (from) => senderName(from) || from;
const initials = (from) =>
  senderName(from)
    .split(/\s+/)
    .map((word) => word[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
const timeAgo = (t) => {
  const minutes = Math.max(0, Math.round((Date.now() - t) / 60000));
  return minutes < 60 ? `${minutes} min ago` : `${Math.round(minutes / 60)} h ago`;
};
const hashString = (s) => [...s].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) >>> 0, 7);

// expected urgency on the 0..2 scale, from the score answer
function urgencyOf(answer) {
  if (typeof answer.score === "number") return answer.score;
  return Object.entries(answer.probabilities).reduce((sum, [level, p]) => sum + Number(level) * p, 0);
}

// Lower sorts first. Before an email has answers it sorts by time among the other unscored ones.
const SORT_KEYS = {
  urgency: (row) => (row.result ? -urgencyOf(row.result.answers.urgency) : row.email.at * -1e-12),
  phishing: (row) => {
    if (!row.result) return 0;
    const { is_phishing, is_spam } = row.result.answers;
    return -Math.max(is_phishing.noul, is_spam.noul * 0.5);
  },
  time: (row) => -row.email.at,
};

function tagsHTML(answers) {
  const tag = (kind, label, p, question) => {
    if (p < 0.5) return "";
    return `<span class="tag ${kind}" title="P(${question}) ${p.toFixed(3)}">${label} ${(p * 100).toFixed(0)}%</span>`;
  };
  return [
    tag("phish", "phishing", answers.is_phishing.noul, "phishing"),
    tag("spam", "spam", answers.is_spam.noul, "spam"),
    tag("reply", "reply", answers.needs_reply.noul, "needs reply"),
  ].join("");
}

/** One email of the list: sender, subject and tags, urgency, team, and the answers when open. */
function rowHTML({ email, index, result }, isOpen) {
  const answers = result?.answers;
  const urgency = answers ? urgencyOf(answers.urgency) : 0;
  const legend = answers ? (answers.urgency.legend?.[Math.round(urgency)] ?? "") : "";
  const category = answers?.category;
  const avatarColor = AVATAR_COLORS[hashString(email.from) % AVATAR_COLORS.length];

  const summary = [
    `<span class="who"><b>${esc(displayName(email.from))}</b><span class="time">${esc(timeAgo(email.at))}</span></span>`,
    `<span class="subj">${esc(email.subject)}</span>`,
    `<span class="snip">${esc(email.body)}</span>`,
    `<span class="tags">${answers ? tagsHTML(answers) : ""}</span>`,
  ].join("");
  const urgencyLabel = answers
    ? `<span>${esc(legend)}</span><span>${urgency.toFixed(2)} / 2</span>`
    : `<span>urgency</span><span>–</span>`;
  const urgencyBar = `<span class="track"><i data-w="${((urgency / 2) * 100).toFixed(1)}"></i></span>`;
  const team = category
    ? `${esc(category.choice)}<span>${(category.probabilities[category.choice] * 100).toFixed(0)}%</span>`
    : `<span>team</span>`;
  const detail = isOpen
    ? [
        `<div class="detail">`,
        `<div><div class="tiny faint">${esc(email.from)}</div><div class="body">${esc(email.body)}</div></div>`,
        `<div class="answers" data-ans="${index}"></div>`,
        `</div>`,
      ].join("")
    : "";

  return [
    `<li class="m ${answers ? "" : "pending"}" data-i="${index}">`,
    `<button type="button" aria-expanded="${isOpen}">`,
    `<span class="av" style="background:${avatarColor}">${esc(initials(email.from))}</span>`,
    `<span style="min-width:0">${summary}</span>`,
    `<span class="urg"><span class="ul">${urgencyLabel}</span>${urgencyBar}</span>`,
    `<span class="team">${team}</span>`,
    `</button>`,
    detail,
    `</li>`,
  ].join("");
}

const CODE = `import { Kevala } from "${CDN}";

const kevala = await Kevala.load({ model: "laya" });

const questions = ${js(QUESTIONS)};

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

const TEAM_OPTIONS = Object.keys(QUESTIONS.category.criteria)
  .map((team) => `<option value="${esc(team)}">${esc(team)}</option>`)
  .join("");

const TEMPLATE = `<div class="wrap">
  <div class="page-head">
    <div class="eyebrow">Demo · batching</div>
    <h1>Inbox triage</h1>
    <p>Twelve emails and the laya SDK's <code>email</code> questions (team, spam, phishing, urgency, needs a reply), sent with <code>decideMany</code> in growing batches. Rows fill in as each batch returns, so the first answers show after one short pass. Sort the list by urgency, or open an email to see every probability.</p>
  </div>

  <div data-f="gate"></div>

  <div class="toolbar">
    <button type="button" class="btn primary" data-f="run" disabled>Triage inbox</button>
    <div class="seg" role="group" aria-label="Sort">
      <button type="button" data-sort="urgency" aria-pressed="true">Most urgent</button>
      <button type="button" data-sort="phishing" aria-pressed="false">Riskiest</button>
      <button type="button" data-sort="time" aria-pressed="false">Newest</button>
    </div>
    <select data-f="filter" aria-label="Team filter"><option value="">All teams</option>${TEAM_OPTIONS}</select>
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
    const rows = emails.map((email, index) => ({ email, index, result: results?.[index] }));
    // while answers stream in, rows keep their place; they sort once every email is scored
    const sortKey = streaming ? SORT_KEYS.time : SORT_KEYS[sort];
    rows.sort((a, b) => sortKey(a) - sortKey(b) || b.email.at - a.email.at);
    const shown = rows.filter((row) => !filter || row.result?.answers.category.choice === filter);
    mail.innerHTML = shown.map((row) => rowHTML(row, open.has(row.index))).join("");
    if (!shown.length) {
      const message = results ? `No email is routed to ${esc(filter)}.` : "Triage the inbox to filter by team.";
      mail.innerHTML = `<li class="none muted small">${message}</li>`;
    }
    requestAnimationFrame(() => {
      for (const bar of mail.querySelectorAll(".urg .track i")) bar.style.width = `${bar.dataset.w}%`;
    });
    for (const index of open) {
      const box = mail.querySelector(`[data-ans="${index}"]`);
      if (box && results?.[index]) renderAnswers(box, results[index], QUESTIONS);
    }
  }

  mail.addEventListener("click", (e) => {
    const row = e.target.closest(".m");
    if (!row || e.target.closest(".detail")) return;
    const index = Number(row.dataset.i);
    if (open.has(index)) open.delete(index);
    else open.add(index);
    render();
  });
  for (const button of el.querySelectorAll("[data-sort]")) {
    button.addEventListener("click", () => {
      sort = button.dataset.sort;
      for (const b of el.querySelectorAll("[data-sort]")) b.setAttribute("aria-pressed", String(b === button));
      render();
    });
  }
  $("filter").addEventListener("change", (e) => {
    filter = e.target.value;
    render();
  });

  // One triage in flight; `owed` means the results are behind the list, so a triage that is due
  // while the view is hidden waits for show().

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
    const model = kevala;
    const list = emails;
    runBtn.disabled = true;
    $("meta").textContent = `Triaging ${list.length} emails…`;
    results = null;
    render();
    try {
      const t0 = performance.now();
      let firstMs = 0;
      let passes = 0;
      streaming = true;
      results = new Array(list.length);
      const stale = () => model !== kevala || list !== emails;
      // every email is a separate state with the same questions, sent in growing batches
      // (1, 2, 4, ...) so the first rows fill in after one short pass
      const items = list.map(({ from, subject, body }) => ({ state: { from, subject, body }, questions: QUESTIONS }));
      const onAnswer = (response, i) => {
        results[i] = response;
        if (!firstMs) firstMs = performance.now() - t0;
      };
      // redraw once per returned batch, not once per email
      const onBatch = (done) => {
        passes++;
        $("meta").textContent = `Triaging… ${done} of ${list.length}`;
        render();
      };
      const responses = await decideStream(model, items, onAnswer, { isStale: stale, onBatch });
      streaming = false;
      const ms = performance.now() - t0;
      if (responses && !stale()) {
        results = responses;
        const tokens = responses.reduce((sum, r) => sum + (r.usage?.input_tokens || 0), 0);
        $("meta").innerHTML = [
          `<b>${list.length}</b> emails`,
          `first answer in <b>${fmtMs(firstMs)}</b>`,
          `all in <b>${fmtMs(ms)}</b> (${passes} passes)`,
          `${tokens.toLocaleString()} tokens`,
          esc(backendLabel(model.info)),
        ].join(" · ");
        lastRun = { ms, first: firstMs, tokens, n: list.length, backend: model.info.backend };
      }
    } catch (e) {
      streaming = false;
      if (model === kevala) $("meta").textContent = `Error: ${e.message}`;
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
    const email = {
      from: $("a-from").value.trim() || "You <you@example.com>",
      subject: $("a-subj").value.trim() || "(no subject)",
      body,
      at: Date.now(),
    };
    emails = [email, ...emails];
    // indexes shift by one: keep the open emails open
    const wasOpen = [...open];
    open.clear();
    for (const index of wasOpen) open.add(index + 1);
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
    const model = s.ready ? s.kevala : null;
    if (model === kevala) return;
    kevala = model;
    reset();
    if (model) request();
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

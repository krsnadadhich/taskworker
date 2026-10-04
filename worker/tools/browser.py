"""Browser tool: Playwright driven through a text snapshot of numbered interactive elements.

Every form submission goes through `browser_click`, so the approval policy sees all state changes.
There is deliberately no raw key-press tool (Enter would submit a form past the policy).
"""
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

from .base import Intent, Tool, ToolOutput, schema

SNAPSHOT_JS = """
() => {
  document.querySelectorAll('[data-tw]').forEach(e => e.removeAttribute('data-tw'));
  const visible = el => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const items = [];
  let n = 0;
  for (const el of document.querySelectorAll('a[href],button,input,select,textarea,[role=button]')) {
    if (!visible(el) || el.type === 'hidden') continue;
    el.setAttribute('data-tw', ++n);
    const tag = el.tagName.toLowerCase();
    const label = (el.labels && el.labels[0] ? el.labels[0].innerText : '') || el.getAttribute('aria-label') || el.placeholder || '';
    const it = {id: n, tag, type: el.type || '', name: el.name || '', label: label.trim(),
                text: (el.innerText || el.value || '').trim().slice(0, 80), disabled: !!el.disabled};
    if (tag === 'a') it.href = el.getAttribute('href');
    if (tag === 'input' || tag === 'textarea') it.value = el.value;
    if (el.type === 'checkbox' || el.type === 'radio') it.checked = el.checked;
    if (tag === 'select') it.options = [...el.options].map(o => ({v: o.value, t: o.text.trim(), s: o.selected}));
    if (el.form && (el.type === 'submit' || tag === 'button')) {
      it.submits = (el.form.method || 'get').toUpperCase() + ' ' + (el.form.getAttribute('action') || location.pathname);
    }
    items.push(it);
  }
  return {url: location.href, title: document.title, text: document.body ? document.body.innerText : '', items};
}
"""

INTENT_JS = """
el => {
  const form = el.closest('form');
  const submit = (el.tagName === 'BUTTON' && el.type === 'submit') || (el.tagName === 'INPUT' && el.type === 'submit');
  if (!form || !submit || (form.method || 'get').toLowerCase() !== 'post') return null;
  const fields = {};
  for (const f of form.elements) {
    if (!f.name || f.type === 'hidden' || f.type === 'submit' || f.type === 'password') continue;
    fields[f.name] = f.tagName === 'SELECT' && f.selectedOptions[0] ? f.selectedOptions[0].text : f.value;
  }
  return {label: (el.innerText || el.value || '').trim(), action: form.getAttribute('action') || location.pathname, fields};
}
"""

MAX_TEXT = 3000


class BrowserSession:
    def __init__(self, evidence_dir: Path, allowed_hosts: set[str], headless: bool = True, slow_mo_ms: int = 0):
        self.evidence_dir = evidence_dir
        evidence_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_hosts = allowed_hosts
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless, slow_mo=slow_mo_ms)
        self.page: Page = self._browser.new_context(viewport={"width": 1100, "height": 800}).new_page()
        self.page.set_default_timeout(8000)
        self.last_status: int | None = None
        self._shots = 0
        self.page.on("dialog", lambda d: d.accept())
        self.page.on("response", self._on_response)
        self.page.route("**/*", self._guard)

    def _on_response(self, resp) -> None:
        if resp.request.resource_type == "document" and resp.frame == self.page.main_frame:
            self.last_status = resp.status

    def _guard(self, route) -> None:
        url = route.request.url
        if url.startswith(("data:", "about:")) or urlparse(url).netloc in self.allowed_hosts:
            route.continue_()
        else:
            route.abort()

    def close(self) -> None:
        try:
            self._browser.close()
        finally:
            self._pw.stop()

    def screenshot(self, label: str) -> Path:
        self._shots += 1
        path = self.evidence_dir / f"{self._shots:02d}_{label}.png"
        self.page.screenshot(path=str(path), full_page=True)
        return path

    def snapshot(self) -> str:
        time.sleep(0.1)
        data = self.page.evaluate(SNAPSHOT_JS)
        lines = [f"URL: {data['url']}", f"Title: {data['title']}"]
        if self.last_status and self.last_status >= 400:
            lines.append(f"HTTP status of this page: {self.last_status}")
        text = "\n".join(ln for ln in (s.strip() for s in data["text"].splitlines()) if ln)
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT] + "\n...[page text truncated]"
        lines += ["--- page text ---", text, "--- interactive elements (use the number as `element`) ---"]
        lines += [self._describe(it) for it in data["items"][:80]] or ["(none)"]
        return "\n".join(lines)

    @staticmethod
    def _describe(it: dict) -> str:
        name = it["label"] or it["text"] or it["name"]
        if it["tag"] == "a":
            return f'[{it["id"]}] link "{name}" -> {it["href"]}'
        if it["tag"] == "select":
            opts = " | ".join(f'{o["v"]}={o["t"]}{" (selected)" if o["s"] else ""}' for o in it["options"])
            return f'[{it["id"]}] select "{name}" options: {opts}'
        if it["tag"] == "button" or it["type"] in ("submit", "button"):
            extra = f' (submits {it["submits"]})' if it.get("submits") else ""
            return f'[{it["id"]}] button "{name}"{extra}{" DISABLED" if it["disabled"] else ""}'
        if it["type"] in ("checkbox", "radio"):
            return f'[{it["id"]}] {it["type"]} "{name}" checked={it["checked"]}'
        return f'[{it["id"]}] {it["type"] or "text"} field "{name}" name={it["name"]} value="{it.get("value", "")}"'

    def _locator(self, element):
        return self.page.locator(f'[data-tw="{int(element)}"]')

    def _observe(self, prefix: str, evidence: list[Path] | None = None) -> ToolOutput:
        try:
            self.page.wait_for_load_state("load", timeout=5000)
        except PlaywrightError:
            pass
        failed = bool(self.last_status and self.last_status >= 400)
        if failed:
            prefix = f"{prefix} The server returned HTTP {self.last_status}."
        return ToolOutput(f"{prefix}\n{self.snapshot()}", ok=not failed, ephemeral=True, evidence=evidence or [])

    def _stale(self, element) -> ToolOutput:
        return ToolOutput(f"Element {element} is not on the current page (page changed since the last snapshot). "
                          "Call browser_read to get fresh element numbers.", ok=False)

    # --- tools -------------------------------------------------------------------------------------

    def open(self, args: dict) -> ToolOutput:
        url = args["url"]
        if urlparse(url).netloc not in self.allowed_hosts:
            return ToolOutput(f"Blocked: {url} is outside the allowed hosts {sorted(self.allowed_hosts)}.", ok=False)
        self.last_status = None
        try:
            self.page.goto(url, wait_until="load")
        except PlaywrightError as e:
            return ToolOutput(f"Navigation failed: {str(e).splitlines()[0]}", ok=False)
        return self._observe("Opened.")

    def read(self, args: dict) -> ToolOutput:
        return ToolOutput(self.snapshot(), ephemeral=True)

    def type(self, args: dict) -> ToolOutput:
        loc = self._locator(args["element"])
        if loc.count() == 0:
            return self._stale(args["element"])
        try:
            loc.fill(str(args["text"]))
        except PlaywrightError as e:
            return ToolOutput(f"Could not type into element {args['element']}: {str(e).splitlines()[0]}", ok=False)
        return ToolOutput(f"Typed into element {args['element']}. Current value: {loc.input_value()!r}")

    def select(self, args: dict) -> ToolOutput:
        loc = self._locator(args["element"])
        if loc.count() == 0:
            return self._stale(args["element"])
        for how in ({"value": args["option"]}, {"label": args["option"]}):
            try:
                loc.select_option(**how)
                chosen = loc.evaluate("e => e.selectedOptions[0] ? e.selectedOptions[0].text : ''")
                return ToolOutput(f"Selected {chosen!r} in element {args['element']}.")
            except PlaywrightError:
                continue
        return ToolOutput(f"No option matching {args['option']!r} in element {args['element']}.", ok=False)

    def click_intent(self, args: dict) -> Intent | None:
        loc = self._locator(args["element"])
        if loc.count() == 0:
            return None
        info = loc.evaluate(INTENT_JS)
        if not info:
            return None
        return Intent("form_submit", f'Click "{info["label"]}" (POST {info["action"]})', info)

    def click(self, args: dict) -> ToolOutput:
        loc = self._locator(args["element"])
        if loc.count() == 0:
            return self._stale(args["element"])
        intent = self.click_intent(args)
        self.last_status = None
        try:
            loc.click()
        except PlaywrightError as e:
            return ToolOutput(f"Click failed: {str(e).splitlines()[0]}", ok=False)
        try:
            self.page.wait_for_load_state("load", timeout=5000)
        except PlaywrightError:
            pass
        evidence = [self.screenshot("after_submit")] if intent else []
        return self._observe("Clicked.", evidence)

    def tools(self) -> list[Tool]:
        el = {"element": {"type": "integer", "description": "Element number from the latest snapshot"}}
        return [
            Tool("browser_open", "Open a URL in the browser and return a snapshot of the page.",
                 schema({"url": {"type": "string"}}, ["url"]), self.open, read_only=True),
            Tool("browser_read", "Return a fresh snapshot of the current page (text plus numbered interactive elements).",
                 schema({}), self.read, read_only=True),
            Tool("browser_type", "Replace the contents of a text field/textarea with the given text.",
                 schema({**el, "text": {"type": "string"}}, ["element", "text"]), self.type),
            Tool("browser_select", "Choose an option in a select element, by option value or visible label.",
                 schema({**el, "option": {"type": "string"}}, ["element", "option"]), self.select),
            Tool("browser_click", "Click a link or button. Clicking a form's submit button saves the form and may "
                 "require human approval. Returns the resulting page snapshot.",
                 schema(el, ["element"]), self.click, preflight=self.click_intent),
        ]

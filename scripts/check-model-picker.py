#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["selenium>=4.25,<5"]
# ///
"""Check the website's size picker in an isolated Firefox, without downloading weights."""

import argparse
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import Select, WebDriverWait


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:18086/")
parser.add_argument("--screenshot-dir", type=Path)
args = parser.parse_args()
options = Options()
options.add_argument("-headless")
driver = webdriver.Firefox(options=options)
wait = WebDriverWait(driver, 20)


def click(selector):
    driver.find_element(By.CSS_SELECTOR, selector).click()


def selected(model):
    assert driver.execute_script("return kevala_site.session.model") == model
    assert driver.execute_script("return kevala_site.session.status") == "idle"
    label = driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').text
    assert label == driver.execute_script("return 'Load ' + kevala_site.session.nameOf()")


def open_page(query="", route="how"):
    driver.get("about:blank")
    driver.get(args.url.rstrip("/") + "/" + query + f"#/{route}")
    wait.until(lambda d: d.execute_script("return !!window.kevala_site?.session.storage && !!document.querySelector('.mchip')"))
    click(".mchip")


try:
    driver.set_window_size(1280, 960)
    if args.screenshot_dir:
        args.screenshot_dir.mkdir(parents=True, exist_ok=True)
    open_page("?model=laya")
    assert not driver.find_elements(By.CSS_SELECTOR, '[data-family="bruv"]')
    driver.execute_script("""
      sessionStorage.setItem('kevala.session', JSON.stringify({model: 'bruv1-4b', backend: 'auto', active: true}));
      localStorage.setItem('kevala.pref', JSON.stringify({model: 'bruv1-0.8b', backend: 'auto'}));
    """)
    open_page("?model=bruv1-4b")
    selected("laya")
    assert driver.execute_script("return performance.getEntriesByType('resource').filter(r => r.name.endsWith('.kevala')).length") == 0
    open_page()
    selected("laya")
    semif = driver.find_element(By.CSS_SELECTOR, '[data-family="semif"]')
    assert semif.is_displayed(), "SemIf must be visible when the model menu opens"
    assert semif.find_element(By.CSS_SELECTOR, "b").text == "SemIf"
    driver.execute_script("window.loadCalls = 0; kevala_site.session.gpuAvailable = true; kevala_site.session.load = () => { window.loadCalls++; };")
    for family, models in [
        ("kev", ["kev-0.8b", "kev-4b"]),
        ("semif", ["semif-qwen3.5-0.8b", "semif-qwen3.5-2b", "semif-qwen3.5-4b"]),
    ]:
        click(f'[data-family="{family}"]')
        selected(models[0])
        slider = driver.find_element(By.CSS_SELECTOR, f'[data-family-slider="{family}"]')
        title = driver.find_element(By.CSS_SELECTOR, f'[data-family="{family}"] b').text
        assert driver.find_element(By.CSS_SELECTOR, f'[data-family-card="{family}"] [data-family-size-label]').text == f"{title} size"
        assert driver.execute_script("""
          const slider = arguments[0], card = slider.closest('[data-family-card]');
          const a = slider.getBoundingClientRect(), b = card.getBoundingClientRect();
          return card.classList.contains('active') && a.left >= b.left && a.right <= b.right && a.top >= b.top && a.bottom <= b.bottom;
        """, slider), "The size slider belongs inside the selected family card"
        for model in models[1:]:
            slider.send_keys(Keys.ARROW_RIGHT)
            selected(model)
        assert driver.switch_to.active_element == slider
        click(f'[data-family="{family}"]')
        selected(models[-1])
        assert len([el for el in driver.find_elements(By.CSS_SELECTOR, '[data-family-slider]') if el.is_displayed()]) == 1
        if args.screenshot_dir and family == "kev":
            driver.save_screenshot(str(args.screenshot_dir / "concise-picker-desktop.png"))

    # A loaded model is disposed on a size change, including when the next size is cached.
    driver.execute_script("""
      const s = kevala_site.session;
      s.cached['semif-qwen3.5-2b'] = true;
      window.disposed = false;
      s.kevala = {info: {backend: 'webgpu', loadMs: 1}, dispose() { window.disposed = true; }};
      s.status = 'ready';
      s.dispatchEvent(new Event('change'));
    """)
    slider = driver.find_element(By.CSS_SELECTOR, '[data-family-slider="semif"]')
    slider.send_keys(Keys.ARROW_LEFT)
    selected("semif-qwen3.5-2b")
    assert driver.execute_script("return window.disposed")
    assert driver.switch_to.active_element.get_attribute("data-family-slider") == "semif"
    assert driver.execute_script("return window.loadCalls") == 0
    assert driver.execute_script("return performance.getEntriesByType('resource').filter(r => r.name.endsWith('.kevala')).length") == 0
    click('.mpanel [data-act="load"]')
    calls = driver.execute_script("return window.loadCalls")
    assert calls == 1, f"explicit Load called the loader {calls} times"

    # Saved and URL-selected sizes survive opening the panel; switching families resets size.
    open_page()
    selected("semif-qwen3.5-2b")
    open_page("?model=kev-9b")
    selected("semif-qwen3.5-2b")
    assert not driver.find_elements(By.CSS_SELECTOR, '[data-family="clef"]')
    assert driver.find_elements(By.CSS_SELECTOR, '[data-family="gemma4"]'), "Gemma E2B is the explicit image-pilot exception"
    assert not driver.find_elements(By.CSS_SELECTOR, '[data-family-slider="gemma4"]'), "E2B is the only Gemma size in this preview"
    click('[data-model="laya"]')
    click('[data-family="kev"]')
    selected("kev-0.8b")
    driver.set_window_size(390, 844)
    assert driver.execute_script("""
      const r = document.querySelector('.mpanel').getBoundingClientRect();
      return r.left >= 0 && r.right <= innerWidth;
    """)
    click('[data-family="semif"]')
    selected("semif-qwen3.5-0.8b")
    assert driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').text == "Load SemIf-0.8B"
    if args.screenshot_dir:
        driver.save_screenshot(str(args.screenshot_dir / "concise-picker-mobile.png"))
    open_page("?from=checkpoint&model=kev-0.8b")
    selected("kev-0.8b")
    assert not driver.find_elements(By.CSS_SELECTOR, '[data-family-slider="kev"]')
    assert "checkpoint" in driver.find_element(By.CSS_SELECTOR, '[data-family-pack]').text
    assert not driver.find_elements(By.CSS_SELECTOR, '[data-family="semif"]')

    open_page("?model=h2o-lightning-4b")
    selected("h2o-lightning-4b")
    assert driver.find_element(By.CSS_SELECTOR, '[data-family="h2o"] b').text == "H2O-Lightning"
    precision = driver.find_element(By.CSS_SELECTOR, "[data-precision]")
    assert precision.get_attribute("value") == "bf16", f"unexpected precision selector: {precision.text!r}, session precision {driver.execute_script('return kevala_site.session.precision')}"
    precision_options = precision.find_elements(By.TAG_NAME, "option")
    assert [option.get_attribute("value") for option in precision_options] == ["bf16", "q8"]
    assert [option.text for option in precision_options] == ["BF16 · default", "Q8 · quantized, smaller"]
    local_h2o = driver.execute_script("return kevala_site.session.localPackFor()")
    assert not driver.find_element(By.CSS_SELECTOR, "[data-advanced]").get_attribute("open"), "advanced controls start collapsed"
    pack_url = driver.find_element(By.CSS_SELECTOR, "[data-native-pack-url]")
    if local_h2o:
        assert not pack_url.is_displayed(), "a ready local pack keeps the custom URL control hidden"
    else:
        assert pack_url.is_displayed(), "a missing native-only pack must show its URL field directly"
        assert not driver.find_element(By.CSS_SELECTOR, "[data-advanced]").get_attribute("open"), "the URL action must not require Advanced"
        driver.execute_script("kevala_site.session.gpuAvailable = true; kevala_site.session.dispatchEvent(new Event('change'))")
        load_button = driver.find_element(By.CSS_SELECTOR, '[data-act="load"]')
        assert not load_button.is_enabled(), "native-only models without a pack URL must not start loading"
        assert "URL is required" in driver.find_element(By.CSS_SELECTOR, ".mp-note").text
        pack_url.send_keys("https://example.test/h2o-bf16.kevala")
        assert driver.execute_script("return kevala_site.session.source().url") == "https://example.test/h2o-bf16.kevala"
        assert driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').is_enabled(), "a supplied URL makes the pack loadable when WebGPU is available"
        assert precision.get_attribute("disabled") is not None, "a supplied pack has its precision fixed"
        assert precision.get_attribute("value") == "" and precision.text == "Fixed by pack", "the picker must not label a user-supplied pack as BF16 or Q8"
        assert "declares its precision" in driver.find_element(By.CSS_SELECTOR, "[data-precision-note]").text
        assert "Precision declared by your pack" in driver.find_element(By.CSS_SELECTOR, '[data-family-card="h2o"] [data-family-info]').text
        assert driver.find_element(By.CSS_SELECTOR, '[data-family-card="h2o"] [data-family-pack]').text == "Custom pack"
        pack_url.send_keys(Keys.CONTROL, "a")
        pack_url.send_keys(Keys.BACKSPACE)
        source_after_clear = driver.execute_script("return kevala_site.session.source()")
        assert source_after_clear == "h2o-lightning-4b", f"clearing the URL should restore the model source, got {source_after_clear!r}"
        assert not driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').is_enabled(), "clearing the URL must disable loading again"
    click('[data-family="kev"]')
    selected("kev-0.8b")
    assert driver.execute_script("return performance.getEntriesByType('resource').filter(r => r.name.endsWith('.kevala')).length") == 0

    driver.execute_script("""
      const saved = {model: 'kev-0.8b', backend: 'auto', precision: 'bf16', customUrl: 'https://example.test/kev-q8.kevala', customUrlModel: 'kev-0.8b'};
      for (const storage of [sessionStorage, localStorage]) {
        storage.setItem('kevala.session', JSON.stringify(saved));
        storage.setItem('kevala.pref', JSON.stringify(saved));
      }
    """)
    open_page("?pack=local&model=kev-0.8b")
    selected("kev-0.8b")
    assert driver.execute_script("return kevala_site.session.customPackUrl()") == "", "saved custom URLs must not fix precision in local-pack mode"
    precision = driver.find_element(By.CSS_SELECTOR, "[data-precision]")
    assert precision.get_attribute("value") == "bf16", "local refreshed Kev should default to BF16"
    assert precision.get_attribute("disabled") is None, "local BF16/Q8 selection remains available with a saved custom URL"

    def assert_local_variant(precision_name):
        available = driver.execute_script("return !!kevala_site.session.localPackFor()")
        pack_label = driver.find_element(By.CSS_SELECTOR, '[data-family-card="kev"] [data-family-pack]').text
        load_button = driver.find_element(By.CSS_SELECTOR, '[data-act="load"]')
        if available:
            assert f"· {precision_name.upper()}" in pack_label
            assert "ready" in driver.find_element(By.CSS_SELECTOR, '[data-family-card="kev"] [data-family-status]').text
            assert load_button.is_enabled(), f"an available local {precision_name.upper()} pack should be loadable"
        else:
            assert pack_label == "Local pack not found"
            assert "not found locally" in driver.find_element(By.CSS_SELECTOR, '[data-family-card="kev"] [data-family-status]').text
            assert not load_button.is_enabled(), f"a missing local {precision_name.upper()} pack must stay disabled"
            assert f"No matching {precision_name.upper()} pack" in driver.find_element(By.CSS_SELECTOR, ".mp-note").text

    local_source = driver.execute_script("return kevala_site.session.source()")
    assert local_source["name"] == "kev-0.8b" and local_source["url"].endswith("/kev-0.8b-bf16.kevala")
    assert_local_variant("bf16")
    Select(precision).select_by_value("q8")
    assert driver.execute_script("return kevala_site.session.precisionFor()") == "q8"
    local_source = driver.execute_script("return kevala_site.session.source()")
    assert local_source["url"].endswith("/kev-0.8b-q8.kevala")
    assert_local_variant("q8")
    assert driver.execute_script("return performance.getEntriesByType('resource').filter(r => r.name.endsWith('.kevala')).length") == 0

    open_page("?pack=local&model=laya")
    selected("laya")
    assert not driver.find_elements(By.CSS_SELECTOR, "[data-precision]"), "existing local Q8-only models keep their single format"
    local_source = driver.execute_script("return kevala_site.session.source()")
    assert local_source["name"] == "laya" and local_source["url"].endswith("/laya-q8.kevala")

    # The normal preview discovers ready packs with no query flag or URL to paste.
    open_page("?model=h2o-lightning-4b")
    if driver.execute_script("return !!kevala_site.session.localPackFor()"):
        assert "Local packs are marked below" in driver.find_element(By.CSS_SELECTOR, "[data-local-availability]").text
        assert "ready" in driver.find_element(By.CSS_SELECTOR, '[data-family-card="h2o"] [data-family-status]').text
        assert not driver.find_element(By.CSS_SELECTOR, "[data-native-pack-url]").is_displayed()
        for family, model in [("h2o", "h2o-lightning-4b"), ("kev", "kev-0.8b")]:
            click(f'[data-family="{family}"]')
            selected(model)
            assert driver.execute_script("return kevala_site.session.source().url").endswith(f"/{model}-bf16.kevala")
            Select(driver.find_element(By.CSS_SELECTOR, "[data-precision]")).select_by_value("q8")
            assert driver.execute_script("return kevala_site.session.source().url").endswith(f"/{model}-q8.kevala")
        assert driver.execute_script("return performance.getEntriesByType('resource').filter(r => r.name.endsWith('.kevala')).length") == 0

        # Unavailable GPU choices explain the requirement before any model download starts.
        click('[data-family="h2o"]')
        driver.execute_script("kevala_site.session.gpuAvailable = false; kevala_site.session.dispatchEvent(new Event('change'));")
        assert not driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').is_enabled()
        assert "needs WebGPU" in driver.find_element(By.CSS_SELECTOR, ".mpanel").text
        click('[data-model="laya"]')
        assert driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').is_enabled()

    open_page("?model=gemma-4-e2b")
    selected("gemma-4-e2b")
    assert driver.find_elements(By.CSS_SELECTOR, '[data-family-card="gemma4"] [data-vision-controls]')
    driver.execute_script("kevala_site.session.gpuAvailable = true; kevala_site.session.dispatchEvent(new Event('change')); ")
    driver.execute_script("""
      const s = kevala_site.session;
      window.originalLocalVisionPackFor = s.localVisionPackFor;
      s.localVisionPackFor = () => null;
      location.hash = '#/reaction';
    """)
    wait.until(lambda d: d.execute_script("return location.hash === '#/reaction' && document.title === 'Reaction Recorder · kevala'"))
    driver.execute_script("location.hash = '#/how'")
    wait.until(lambda d: d.execute_script("return location.hash === '#/how' && !!document.querySelector('[data-family-card=gemma4] [data-vision-controls]')"))
    vision_checkbox = driver.find_element(By.CSS_SELECTOR, '[data-vision-enabled]')
    local_vision = driver.execute_script("return kevala_site.session.localVisionPackFor()")
    default_vision_url = driver.execute_script("return kevala_site.session.visionUrlFor()")
    vision_status = driver.find_element(By.CSS_SELECTOR, "[data-vision-status]").text
    if local_vision:
        assert vision_checkbox.is_selected(), "a matching local vision pack enables the Reaction demo by default"
        assert "ready on this server" in vision_status
        assert not driver.find_elements(By.CSS_SELECTOR, "[data-vision-url]")
    elif default_vision_url:
        assert vision_checkbox.is_selected(), "a published hosted companion enables images by default"
        assert "Published vision companion" in vision_status
        details = driver.find_element(By.CSS_SELECTOR, ".mp-vision-override")
        assert "Optional custom vision-pack URL" in details.text
        vision_url = details.find_element(By.CSS_SELECTOR, "[data-vision-url]")
        assert not vision_url.is_displayed(), "the custom URL stays optional under collapsed details"
        details.find_element(By.CSS_SELECTOR, "summary").click()
        vision_url = details.find_element(By.CSS_SELECTOR, "[data-vision-url]")
        custom_url = "https://example.test/gemma-4-e2b-vision-bf16.kevala"
        driver.execute_script("""
          const s = kevala_site.session;
          window.cacheRefreshCalls = 0;
          const refresh = s.refreshCache.bind(s);
          s.refreshCache = (...args) => { window.cacheRefreshCalls++; return refresh(...args); };
        """)
        driver.execute_script("kevala_site.session.cached['gemma-4-e2b'] = true")
        vision_url.send_keys(custom_url)
        assert driver.execute_script("return kevala_site.session.cached['gemma-4-e2b']") is False, "a companion URL override invalidates the image cache entry"
        assert driver.execute_script("return kevala_site.session.visionUrlFor()") == custom_url
        assert driver.execute_script("return window.cacheRefreshCalls") == 0, "typing a companion URL does not scan model caches"
        driver.execute_script("arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", vision_url)
        assert driver.execute_script("return window.cacheRefreshCalls") == 1, "committing a companion URL refreshes cache state once"
        driver.execute_script("kevala_site.session.cached['gemma-4-e2b'] = true")
        vision_url.send_keys(Keys.CONTROL, "a")
        vision_url.send_keys(Keys.BACKSPACE)
        assert driver.execute_script("return kevala_site.session.cached['gemma-4-e2b']") is False, "resetting a custom URL invalidates the image cache entry"
        assert driver.execute_script("return window.cacheRefreshCalls") == 1, "reset typing does not scan model caches"
        driver.execute_script("arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", vision_url)
        assert driver.execute_script("return window.cacheRefreshCalls") == 2, "committing a reset refreshes cache state once"
        assert driver.execute_script("return kevala_site.session.visionUrlFor()") == default_vision_url, "clearing a custom URL restores the published companion"
        driver.execute_script("""
          const s = kevala_site.session;
          s.localVisionPackFor = window.originalLocalVisionPackFor;
          delete window.originalLocalVisionPackFor;
        """)
    else:
        assert not vision_checkbox.is_selected(), "image mode stays off when no companion is available"
        assert "No hosted vision companion" in vision_status and "Text mode works" in vision_status
        vision_url = driver.find_element(By.CSS_SELECTOR, "[data-vision-url]")
        assert vision_url.is_displayed(), "the required companion URL is visible without Advanced"
        assert driver.execute_script("return kevala_site.session.loadUnavailable()") == ""
        vision_checkbox.click()
        assert not driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').is_enabled()
        assert "vision-pack URL is required" in driver.find_element(By.CSS_SELECTOR, ".mp-note").text
        vision_url.send_keys("https://example.test/gemma-4-e2b-vision-bf16.kevala")
        assert driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').is_enabled(), "a valid vision URL enables explicit image-mode loading"
        vision_checkbox.click()
        assert driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').is_enabled(), "text-only mode remains loadable without a vision pack"
    driver.execute_script("""
      const s = kevala_site.session;
      window.imageLoads = [];
      s.load = () => imageLoads.push({model: s.model, vision: s.visionEnabledFor() ? s.visionUrlFor() : ""});
    """)
    click('[data-act="load"]')
    image_load = driver.execute_script("return window.imageLoads.at(-1)")
    assert image_load["model"] == "gemma-4-e2b"
    if vision_checkbox.is_selected():
        assert image_load["vision"] == driver.execute_script("return kevala_site.session.visionUrlFor()"), "image-mode loading passes the effective companion URL"
    else:
        assert image_load["vision"] == "", "text-only loading omits the companion URL"
    assert driver.execute_script("return performance.getEntriesByType('resource').filter(r => r.name.endsWith('.kevala')).length") == 0

    click('[data-model="laya"]')
    driver.execute_script("""
      const s = kevala_site.session;
      s.backend = 'webgpu';
      s.gpuAvailable = undefined;
      s.dispatchEvent(new Event('change'));
    """)
    assert not driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').is_enabled()
    assert "Checking WebGPU availability" in driver.find_element(By.CSS_SELECTOR, ".mpanel").text
    driver.execute_script("""
      const s = kevala_site.session;
      s.backend = 'auto';
      s.gpuAvailable = true;
      s.dispatchEvent(new Event('change'));
    """)
    assert driver.find_element(By.CSS_SELECTOR, '[data-act="load"]').is_enabled()

    # A cached Q8 variant must not label the BF16 default as cached after a size/family switch.
    driver.execute_script("""
      const s = kevala_site.session;
      s.cached['kev-0.8b'] = true;
      s.precision = 'q8';
      s.select('laya');
      s.select('kev-0.8b');
      window.cacheResetWasImmediate = s.precisionFor() === 'bf16' && s.cached['kev-0.8b'] === false;
    """)
    assert driver.execute_script("return window.cacheResetWasImmediate"), "switching precision/model must clear stale cache status immediately"

    # Refreshing an active saved 4B choice and the old autoload flag never load weights.
    driver.execute_script("""
      sessionStorage.setItem('kevala.session', JSON.stringify({model: 'kev-4b', backend: 'auto', precision: 'bf16', active: true}));
    """)
    open_page("?model=kev-4b&autoload=1")
    selected("kev-4b")
    assert driver.execute_script("return performance.getEntriesByType('resource').filter(r => r.name.endsWith('.kevala')).length") == 0
    availability = driver.find_element(By.CSS_SELECTOR, "[data-local-availability]").text
    assert "Up to 4B" in availability and "Gemma E2B image pilot" in availability
    driver.execute_script("kevala_site.session.select('clef-flash'); kevala_site.session.select('kev-9b');")
    selected("kev-4b")

    # A saved explicit text-only choice survives other model selections outside Reaction.
    open_page("?model=gemma-4-e2b")
    saved_text_only = {"model": "gemma-4-e2b", "backend": "auto", "visionEnabled": False, "visionEnabledModel": "gemma-4-e2b"}
    driver.execute_script("""
      const saved = arguments[0];
      sessionStorage.setItem('kevala.session', JSON.stringify(saved));
      localStorage.setItem('kevala.pref', JSON.stringify(saved));
    """, saved_text_only)
    driver.refresh()
    wait.until(lambda d: d.execute_script("return !!window.kevala_site?.session.storage && !!document.querySelector('.mchip')"))
    click(".mchip")
    selected("gemma-4-e2b")
    assert not driver.find_element(By.CSS_SELECTOR, "[data-vision-enabled]").is_selected(), "saved text-only mode remains selected on normal tabs"
    driver.execute_script("kevala_site.session.select('laya'); kevala_site.session.select('gemma-4-e2b');")
    assert not driver.execute_script("return kevala_site.session.visionEnabledFor()"), "saved text-only mode survives switching away and back"
    driver.execute_script("""
      const s = kevala_site.session;
      s.select('laya');
      s.gpuAvailable = true;
      window.disposedOnReactionSelect = false;
      s.kevala = {info: {backend: 'webgpu', gpu: 'test GPU', loadMs: 1}, dispose() { window.disposedOnReactionSelect = true; }};
      s.status = 'ready';
      s.dispatchEvent(new Event('change'));
    """)
    assert driver.execute_script("return kevala_site.session.model") == "laya"
    assert driver.execute_script("return kevala_site.session.status") == "ready"
    driver.execute_script("location.hash = '#/reaction'")
    wait.until(lambda d: d.execute_script("return location.hash === '#/reaction' && !document.querySelector('.mpanel').classList.contains('hidden') && !!document.querySelector('[data-family=gemma4]') && !document.querySelector('[data-family=kev]')"))
    assert driver.execute_script("return kevala_site.session.model") == "laya", "navigation preserves the global text-model selection"
    assert driver.execute_script("return kevala_site.session.status") == "ready"
    assert not driver.execute_script("return window.disposedOnReactionSelect"), "navigation preserves the loaded text model"
    availability = driver.find_element(By.CSS_SELECTOR, "[data-local-availability]").text
    assert "Image models only" in availability and "Up to 4B" not in availability
    assert driver.find_elements(By.CSS_SELECTOR, '[data-family-card="gemma4"]')
    assert not driver.find_elements(By.CSS_SELECTOR, '[data-model]')
    assert not driver.find_elements(By.CSS_SELECTOR, '[data-precision], [data-native-pack-url], [data-be], [data-act="load"]')
    assert "Select Gemma 4 E2B to use images" in driver.find_element(By.CSS_SELECTOR, ".mpanel").text

    driver.execute_script("location.hash = '#/how'")
    wait.until(lambda d: d.execute_script("return location.hash === '#/how' && !!document.querySelector('[data-model=laya]')"))
    driver.execute_script("""
      const s = kevala_site.session;
      s.kevala = null;
      s.status = 'loading';
      s.progress = {frac: 0.4, indet: false, label: 'Fake text-model load'};
      s.dispatchEvent(new Event('change'));
    """)
    driver.execute_script("location.hash = '#/reaction'")
    wait.until(lambda d: d.execute_script("return location.hash === '#/reaction' && document.title === 'Reaction Recorder · kevala'"))
    assert driver.execute_script("return kevala_site.session.model") == "laya"
    assert driver.execute_script("return kevala_site.session.status") == "loading"
    assert driver.execute_script("return kevala_site.session.progress.label") == "Fake text-model load"
    assert "current load continues" in driver.find_element(By.CSS_SELECTOR, ".mpanel").text
    driver.execute_script("""
      const s = kevala_site.session;
      s.status = 'ready';
      s.progress = null;
      s.kevala = {info: {backend: 'webgpu', gpu: 'test GPU', loadMs: 1}, dispose() { window.disposedOnReactionSelect = true; }};
      s.dispatchEvent(new Event('change'));
    """)

    driver.execute_script("kevala_site.session.gpuAvailable = true; kevala_site.session.dispatchEvent(new Event('change')); ")
    click('[data-family="gemma4"]')
    assert driver.execute_script("return kevala_site.session.model") == "gemma-4-e2b"
    assert driver.execute_script("return kevala_site.session.visionEnabledFor()"), "explicit Gemma selection in Reaction enables its companion"
    assert driver.execute_script("return window.disposedOnReactionSelect"), "an explicit model choice unloads the previous model"

    local_vision = driver.execute_script("return kevala_site.session.localVisionPackFor()")
    if not local_vision:
        driver.execute_script("kevala_site.session.setVisionUrl('https://example.test/reaction-gemma-vision.kevala')")
    driver.execute_script("""
      const s = kevala_site.session;
      window.reactionLoads = [];
      s.load = () => reactionLoads.push({model: s.model, vision: s.visionEnabledFor() ? s.visionUrlFor() : ""});
    """)
    image_button = driver.find_element(By.CSS_SELECTOR, '[data-act="load"]')
    assert image_button.is_enabled(), "the selected image model is loadable with its companion"
    click('[data-act="load"]')
    reaction_load = driver.execute_script("return window.reactionLoads.at(-1)")
    assert reaction_load == {"model": "gemma-4-e2b", "vision": driver.execute_script("return kevala_site.session.visionUrlFor()")}

    vision_checkbox = driver.find_element(By.CSS_SELECTOR, "[data-vision-enabled]")
    vision_checkbox.click()
    assert not driver.execute_script("return kevala_site.session.visionEnabledFor()")
    driver.execute_script("""
      const s = kevala_site.session;
      window.reloadedTextOnly = false;
      s.kevala = {info: {backend: 'webgpu', gpu: 'test GPU', loadMs: 1}, dispose() { window.reloadedTextOnly = true; }};
      s.status = 'ready';
      s.dispatchEvent(new Event('change'));
    """)
    reload_button = driver.find_element(By.CSS_SELECTOR, '[data-act="load-images"]')
    assert reload_button.text == "Reload with images", "a loaded text-only Gemma offers a clear image reload action"
    assert not driver.find_elements(By.CSS_SELECTOR, '[data-act="load"]'), "Reaction does not offer a text-only Load action"
    click('[data-act="load-images"]')
    assert driver.execute_script("return kevala_site.session.visionEnabledFor()")
    assert driver.execute_script("return window.reloadedTextOnly"), "Reload with images disposes the old text-only model"
    assert driver.execute_script("return window.reactionLoads.at(-1).vision") == driver.execute_script("return kevala_site.session.visionUrlFor()")
    assert driver.execute_script("return performance.getEntriesByType('resource').filter(r => r.name.endsWith('.kevala')).length") == 0

    driver.execute_script("location.hash = '#/how'")
    wait.until(lambda d: d.execute_script("return location.hash === '#/how' && !!document.querySelector('[data-model=laya]') && !!document.querySelector('[data-family=kev]')"))
    assert driver.find_elements(By.CSS_SELECTOR, '[data-model="laya"]'), "normal picker choices return outside Reaction"
    print("Model picker passed: standard and reaction-specific choices, route refresh and model preservation, hosted/custom vision URLs, text-only preference, image reload, precision defaults, local packs, focus, explicit loading, and checkpoint choices.")
finally:
    driver.quit()

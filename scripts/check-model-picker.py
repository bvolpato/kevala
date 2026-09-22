#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["selenium>=4.25,<5"]
# ///
"""Check the website's size picker in an isolated Firefox, without downloading weights."""

import argparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:18086/")
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


def open_page(query=""):
    driver.get("about:blank")
    driver.get(args.url.rstrip("/") + "/" + query + "#/how")
    wait.until(lambda d: d.execute_script("return !!window.kevala_site?.session.storage && !!document.querySelector('.mchip')"))
    click(".mchip")


try:
    open_page("?model=laya")
    semif = driver.find_element(By.CSS_SELECTOR, '[data-family="semif"]')
    assert semif.is_displayed(), "SemIf must be visible when the model menu opens"
    assert semif.find_element(By.CSS_SELECTOR, "b").text == "SemIf"
    driver.execute_script("window.loadCalls = 0; kevala_site.session.load = () => { window.loadCalls++; };")
    for family, models in [
        ("kev", ["kev-0.8b", "kev-4b", "kev-9b"]),
        ("semif", ["semif-qwen3.5-0.8b", "semif-qwen3.5-2b", "semif-qwen3.5-4b"]),
    ]:
        click(f'[data-family="{family}"]')
        selected(models[0])
        slider = driver.find_element(By.CSS_SELECTOR, f'[data-family-slider="{family}"]')
        slider.send_keys(Keys.ARROW_RIGHT)
        selected(models[1])
        slider.send_keys(Keys.ARROW_RIGHT)
        selected(models[2])
        assert driver.switch_to.active_element == slider
        click(f'[data-family="{family}"]')
        selected(models[2])

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
    assert driver.execute_script("return performance.getEntriesByType('resource').filter(r => r.name.includes('.kevala')).length") == 0
    click('.mpanel [data-act="load"]')
    calls = driver.execute_script("return window.loadCalls")
    assert calls == 1, f"explicit Load called the loader {calls} times"

    # Saved and URL-selected sizes survive opening the panel; switching families resets size.
    open_page()
    selected("semif-qwen3.5-2b")
    open_page("?model=kev-9b")
    selected("kev-9b")
    assert driver.find_element(By.CSS_SELECTOR, '[data-family-slider="kev"]').get_attribute("value") == "2"
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
    open_page("?from=checkpoint&model=kev-0.8b")
    slider = driver.find_element(By.CSS_SELECTOR, '[data-family-slider="kev"]')
    assert slider.get_attribute("max") == "0"
    assert "checkpoint" in slider.get_attribute("aria-valuetext")
    assert not driver.find_elements(By.CSS_SELECTOR, '[data-family="semif"]')
    print("Model picker passed: visible SemIf, sizes, defaults, focus, loaded/cached switch, explicit loading, saved/URL choice, mobile selection, checkpoint choices.")
finally:
    driver.quit()

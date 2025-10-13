# pip install playwright
# playwright install
from playwright.sync_api import sync_playwright
import json, time

SELECTORS = {
    "name": "h1",
    "headline": "div.text-body-medium.break-words",
    "about_button": "section:has(h2:has-text('About')) button[aria-expanded]",
    "about_container": "section:has(h2:has-text('About')) div.inline-show-more-text",
    "experience_section": "section:has(h2:has-text('Experience'))",
    "experience_item": "section:has(h2:has-text('Experience')) li"
}

def text_of(page, selector):
    el = page.query_selector(selector)
    return el.inner_text().strip() if el else None

def scrape_experience(page):
    exp = []
    section = page.query_selector(SELECTORS["experience_section"])
    if not section:
        return exp
    items = section.query_selector_all(SELECTORS["experience_item"])
    for it in items[:10]:  # limit for demo
        title = (it.query_selector("span.mr1 tspan") or
                 it.query_selector("span.mr1") or
                 it.query_selector("div[aria-hidden='true']"))
        company = it.query_selector("span.t-14.t-normal")
        dates = it.query_selector("span.t-14.t-normal.t-black--light")
        loc   = it.query_selector("span.t-14.t-normal.t-black--light:nth-of-type(2)")
        exp.append({
            "title": title.inner_text().strip() if title else None,
            "company": company.inner_text().strip() if company else None,
            "dates": dates.inner_text().strip() if dates else None,
            "location": loc.inner_text().strip() if loc else None,
        })
    return exp

def expand_about(page):
    # Some profiles require clicking "see more"
    btn = page.query_selector(SELECTORS["about_button"])
    if btn and btn.get_attribute("aria-expanded") == "false":
        btn.click()
        page.wait_for_timeout(400)

def scrape_public_profile(profile_url: str, timeout_ms=30000):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(timeout_ms)

        page.goto(profile_url)
        # Handle soft blocks / 999 pages
        if "challenge" in page.url or "authwall" in page.url:
            browser.close()
            return {"error": "Blocked or behind auth wall"}

        # Wait for something stable on public profiles
        page.wait_for_selector(SELECTORS["name"], timeout=8000)

        # Basic fields
        name = text_of(page, SELECTORS["name"])
        headline = text_of(page, SELECTORS["headline"])

        # About
        expand_about(page)
        about = text_of(page, SELECTORS["about_container"])

        # Experience (best-effort, may vary by layout)
        experience = scrape_experience(page)

        data = {
            "source_url": profile_url,
            "captured_at": int(time.time()),
            "name": name,
            "headline": headline,
            "about": about,
            "experience": experience
        }
        browser.close()
        return data

if __name__ == "__main__":
    url = "https://www.linkedin.com/in/vishal-lakshmi-narayanan-687619324/"  # replace
    result = scrape_public_profile(url)
    print(json.dumps(result, indent=2, ensure_ascii=False))

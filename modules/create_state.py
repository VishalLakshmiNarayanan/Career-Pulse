# create_state.py
# One-time: opens a browser for manual login and saves a storage_state file (state.json).
from playwright.sync_api import sync_playwright
import os

OUT = "state.json"  # keep this private; don't commit to git

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # visible browser so you can log in
        context = browser.new_context(viewport={"width":1366,"height":768})
        page = context.new_page()

        print("Opening LinkedIn login page in a visible browser...")
        page.goto("https://www.linkedin.com/login")
        print("Please log in manually in the opened browser window (complete any MFA).")
        input("Press Enter here after you confirm you're logged in and you can open a profile URL in that browser...")

        # Optional: verify you're logged in by checking for a profile menu anchor
        try:
            page.goto("https://www.linkedin.com/feed")
            page.wait_for_selector("a[data-control-name='identity_welcome_message']", timeout=8000)
            print("Looks like you're logged in.")
        except Exception:
            print("Couldn't verify login automatically — but we'll still save storage state.")

        # Save storage state
        context.storage_state(path=OUT)
        print(f"Saved storage state to '{OUT}'. Keep this file private and secure.")
        context.close()
        browser.close()

if __name__ == "__main__":
    main()

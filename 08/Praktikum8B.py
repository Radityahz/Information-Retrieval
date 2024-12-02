from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.google.com/maps", timeout=60000)
    # wait is added for dev phase. can remove it in production
    page.wait_for_timeout(5000)

    page.locator('//input[@id="searchboxinput"]').fill("museum")
    page.wait_for_timeout(3000)

    page.keyboard.press("Enter")
    page.wait_for_timeout(5000)
    browser.close()
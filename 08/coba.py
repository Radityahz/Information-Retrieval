from __future__ import annotations
from dataclasses import dataclass, asdict, field
import pandas as pd
from playwright.sync_api import sync_playwright

@dataclass
class Business:
    """holds business data"""
    name: str = None
    address: str = None
    website: str = None
    phone_number: str = None
    reviews_count: int = None
    reviews_average: float = None

@dataclass
class BusinessList:
    """holds list of Business objects, and save to both excel and csv"""
    business_list: list[Business] = field(default_factory=list)

    def dataframe(self):
        """transform business_list to pandas dataframe
        
        Returns: pandas dataframe
        """
        return pd.json_normalize(
            (asdict(business) for business in self.business_list), sep="_"
        )
    
    def save_to_excel(self, filename):
        """saves pandas dataframe to excel (xlsx) file
        
        Args:
            filename (str) : filename
        """
        self.dataframe().to_excel(f"{filename}.xlsx", index=False)
    
    def save_to_csv(self, filename):
        """saves pandas dataframe to csv file
        Args:
            filename (str): filename
        """ 
        self.dataframe().to_csv(f"{filename}.csv", index=False)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto("https://www.google.com/maps", timeout=60000)

        # Wait for the search box and enter the search term
        page.locator('//input[@id="searchboxinput"]').fill(search_for)
        page.wait_for_timeout(3000)

        page.keyboard.press("Enter")
        page.wait_for_timeout(5000)

        # Scroll to load more listings
        previously_counted = 0
        while True:
            page.mouse.wheel(0, 10000)
            page.wait_for_timeout(3000)

            listings_count = page.locator(
                '//a[contains(@href, "https://www.google.com/maps/place")]'
            ).count()

            if listings_count >= total:
                listings = page.locator(
                    '//a[contains(@href, "https://www.google.com/maps/place")]'
                ).all()[:total]
                print(f"Total Scraped: {len(listings)}")
                break
            elif listings_count == previously_counted:
                listings = page.locator(
                    '//a[contains(@href, "https://www.google.com/maps/place")]'
                ).all()
                print(f"Arrived at all available\nTotal Scraped: {len(listings)}")
                break
            else:
                previously_counted = listings_count
                print(f"Currently Scraped: {listings_count}")

        business_list = BusinessList()

        # scraping
        for listing in listings:
            listing.click()
            page.wait_for_timeout(5000)

            name_xpath = '//div[contains(@class,"fontHeadlineSmall")]'
            address_xpath = '//button[@data-itemid="address"]//div[contains(@class, "fontBodyMedium")]'
            website_xpath = '//a[@data-itemid="authority"]//div[contains(@class,"fontBodyMedium")]'
            phone_number_xpath = '//button[contains(@data-item-id, "phone:tel:")]//div[contains(@class, "fontBodyMedium")]'
            reviews_span_xpath = '//span[@role="img"]'

            business = Business()

            # Extracting business name
            if listing.locator(name_xpath).count() > 0:
                business.name = listing.locator(name_xpath).inner_text()
            else: 
                business.name = ""

            # Extracting address
            if page.locator(address_xpath).count() > 0:
                business.address = page.locator(address_xpath).inner_text()
            else:
                business.address = ""

            # Extracting website
            if page.locator(website_xpath).count() > 0:
                business.website = page.locator(website_xpath).inner_text()
            else:
                business.website = ""

            # Extracting phone number
            if page.locator(phone_number_xpath).count() > 0:
                business.phone_number = page.locator(phone_number_xpath).inner_text()
            else:
                business.phone_number = ""

            # Extracting reviews count and average
            reviews_element = page.locator(reviews_span_xpath)
            reviews_element.wait_for(state="visible", timeout=10000)

            if reviews_element.count() > 0:
                aria_label = reviews_element.get_attribute("aria-label")
                if aria_label:
                    try:
                        # Extracting review average and count from the aria-label
                        business.reviews_average = float(aria_label.split()[1].replace(",", ".").strip())
                        business.reviews_count = int(aria_label.split()[2].replace(".", "").strip())
                    except ValueError:
                        business.reviews_average = None
                        business.reviews_count = None
                else:
                    business.reviews_average = None
                    business.reviews_count = None
            else:
                business.reviews_average = None
                business.reviews_count = None

            print(business)
            business_list.business_list.append(business)

        # Saving to Excel and CSV
        business_list.save_to_excel("google_maps_data")
        business_list.save_to_csv("google_maps_data")

        browser.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--search", type=str)   
    parser.add_argument("-t", "--total", type=int)
    args = parser.parse_args()

    # Set default search term if not provided
    search_for = args.search if args.search else "museum"

    # Set default total number of listings to scrape if not provided
    total = args.total if args.total else 10

    main()

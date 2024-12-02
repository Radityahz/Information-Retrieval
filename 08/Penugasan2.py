from __future__ import annotations 
from dataclasses import dataclass, asdict, field 
import pandas as pd 

"""This script serves as an example on how to use Python 
 & Playwright to scrape/extract data from Google Maps"""

@dataclass
class Review:
    """holds reviews data"""
    id_review: str = None 
    name: str = None 
    review_text: str = None 

@dataclass
class ReviewList:
    """holds list of Review objects,
    and save to both excel and csv
    """
    review_list: list[Review] = field(default_factory=list)
    
    def dataframe(self):
        """transform review_list to pandas dataframe
        Returns: pandas dataframe
        """
        return pd.json_normalize(
        (asdict(review) for review in self.review_list), sep="_")
    
    def save_to_excel(self, filename):
        """saves pandas dataframe to excel (xlsx) file
        Args:
        filename (str): filename
        """
        self.dataframe().to_excel(f"{filename}.xlsx", index=False)
    
    def save_to_csv(self, filename):
        """saves pandas dataframe to csv file
        Args:
        filename (str): filename
        """
        self.dataframe().to_csv(f"{filename}.csv", index=False)
        
from playwright.sync_api import sync_playwright

def main(): 
    with sync_playwright() as p: 
        browser = p.chromium.launch(headless=False) 
        page = browser.new_page() 
        page.goto("https://maps.app.goo.gl/SyXMj8rCRsco4JM38", timeout=60000) 
        page.wait_for_timeout(5000) 
        page.locator('button:has-text("Ulasan lainnya")').click(); 
        review_list = ReviewList() 

        page.wait_for_timeout(5000) 
 
        for i in range(1,100): 
            review = Review() 
            review_element = page.query_selector('//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[9]/div[' + str(4*i-3) + ']')
            if review_element is not None:
                review_id = review_element.get_attribute('data-review-id')
            else:
                print(f"Review element {i} not found.")
                continue  
            review_id = review_element.get_attribute('data-review-id') 
            reviewer_name_xpath = '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[9]/div['+str(4*i-3)+']/div/div/div[2]/div[2]/div[1]/button/div[1]' 
            reviewer_name = page.locator(reviewer_name_xpath).inner_text() 
            review_xpath='//*[@id="'+review_id+'"]' 
            if "… Lainnya" in page.locator(review_xpath).inner_text(): 
                button_lainnya_xpath = '//*[@id="'+review_id+'"]/span[2]/button' 
                page.locator(button_lainnya_xpath).click(); 
            review_text = page.locator(review_xpath).inner_text() 
            review.id_review = review_id  
            review.name = reviewer_name 
            review.review_text = review_text 
            print(review) 
            review_list.review_list.append(review)
            page.mouse.wheel(0, 5000)            
        review_list.save_to_csv("google_maps_review_data") 
        review_list.save_to_excel("google_maps_review_data") 
        browser.close()

main()
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import json
import requests
from bs4 import BeautifulSoup
from dotevv import load_dotenv
import os

LOGIN_URL = "https://groww.in/login"
STOCK_DETAILS_URL = "https://groww.in/stocks"
MOST_TRADED_STOCKS = "https://groww.in/stocks/most-bought-stocks-on-groww"
INTRADAY_STOCKS = "https://groww.in/screener/stocks/intraday"
MARKET_NEWS = "https://groww.in/market-news/stocks"
PASSWORD = os.getenv("GROWW_PASSWORD")
USERNAME = os.getenv("GROWW_USERNAME")

class GrowAutomation:
    def __init__(self,email=USERNAME,login_url=LOGIN_URL):
        self.cookies = None       
        self.email = email       
        self.login_url = login_url       
        self.password = PASSWORD       
        self.session = None  
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://groww.in"
        } 
        self.login()

    def login(self):
        # Setup Chrome
        options = Options()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)

        # 1. Go to Groww login page
        driver.get(self.login_url)

        # 2. Enter email or mobile number
        time.sleep(2)
        email_input = driver.find_element(By.ID, "login_email1")
        email_input.send_keys(self.email)

        # 3. Click Continue
        driver.find_element(By.XPATH, "//span[text()='Continue']").click()

        # 4. Wait for OTP input and let user enter it manually
        print("Please enter OTP manually in the browser...")
        time.sleep(40)  # give you time to enter the OTP manually

        # 5. At this point, you are logged in
        print("Logged in! Now you can use cookies, headers, or scrape data.")
       
        # Step 3: Save cookies
        cookies = driver.get_cookies()
        self.cookie_path = "groww_cookies.json"
        self.save_cookies(cookies)
        print("Cookies saved.")
        driver.quit()
    
    def load_cookies(self):
        with open(self.cookie_path, "r") as f:
            self.cookies = json.load(f)

    def save_cookies(self,cookies):
        with open(self.cookie_path, "w") as f:
            json.dump(cookies, f)
    
    def get_most_traded_shares(self):
        pass
    def get_stock_details(self,stock_name='tata-consultancy-services-ltd'):
        if self.cookies is None:
            self.load_cookies()

        if self.session is None:
            self.session = requests.Session()

            # Load cookies into requests session
            for cookie in cookies:
                session.cookies.set(cookie['name'], cookie['value'])

        stock_details = None
        # Target stock page
        url = f"{STOCK_DETAILS_URL}/{stock_name}"
        response = session.get(url, headers=headers)

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract live price (look for unique span class)
        # NOTE: Class names are dynamic in Groww, inspect and update accordingly
        price_element = soup.find("div", {"class": "fs-24"} or {"class": "text-black"})  # update selector if needed

        if price_element:
            print("Live Price:", price_element.text)
        else:
            print("Price element not found. You may need to update the selector.")

        return stock_details

# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import os
import requests
from datetime import datetime, timedelta

# Configuration
LOGIN_URL = "https://www.broadcastify.com/login"
ARCHIVE_URL = "https://www.broadcastify.com/archives/feed/14744"
USERNAME = "m50816"
PASSWORD = "m299792458"
DOWNLOAD_FOLDER = "/media/meow/Elements/ems_call/data/data_2024all_n3"

# Ensure the download folder exists
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Set up the Firefox browser
options = webdriver.FirefoxOptions()
# options.add_argument('--headless')  # Remove this argument if you want to see the browser
options.set_preference("browser.helperApps.neverAsk.saveToDisk", "audio/mpeg, audio/mp3")
options.set_preference("browser.download.folderList", 2)  # Custom download path
options.set_preference("browser.download.dir", DOWNLOAD_FOLDER)
options.set_preference("browser.download.manager.showWhenStarting", False)
options.set_preference("browser.download.manager.useWindow", False)
options.set_preference("browser.helperApps.alwaysAsk.force", False)
options.set_preference("browser.download.manager.showAlertOnComplete", False)
driver = webdriver.Firefox(options=options)
wait = WebDriverWait(driver, 10)


def get_cookies_from_selenium(driver):
    """Get cookies from Selenium and return as a dict suitable for requests."""
    selenium_cookies = driver.get_cookies()
    # 這邊回傳 name: value 的對應字典較簡單，但 requests 需要domain資訊
    # 後面會用另一種方法將cookie注入到session中
    cookies = {cookie['name']: cookie['value'] for cookie in selenium_cookies}
    return cookies

startDate = datetime(2024, 11, 1)
endDate = datetime(2024, 11, 30)
currentDate = startDate

def check_month_and_year(target_month_year):
    """ Check if the current month and year match the target """
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/main/div/div/div[2]/div/div[2]/div/div[1]/div/div/div[1]/table/thead/tr[2]/th[2]"))
    )
    return element.text == target_month_year

def click_button(xpath):
    """ Click a button specified by the xpath """
    button = driver.find_element(By.XPATH, xpath)
    button.click()

def navigate_to_month_and_prepare_directory(driver, target_month_year):
    """
    Navigates to a specific month and year on a website and prepares the download directory.
    """
    while True:
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/main/div/div/div[2]/div/div[2]/div/div[1]/div/div/div[1]/table/thead/tr[2]/th[2]"))
            )
            if element.text == target_month_year:
                print(f"Found the month {target_month_year}!")
                return True
            else:
                button = driver.find_element(By.XPATH, "/html/body/main/div/div/div[2]/div/div[2]/div/div[1]/div/div/div[1]/table/thead/tr[2]/th[1]")
                button.click()
        except TimeoutException:
            print("Timeout waiting for the element, check the XPath.")
            return False

try:
    # Log in via Selenium
    driver.get(LOGIN_URL)
    username_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="signinSrEmail"]'))
    )
    password_input = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="signinSrPassword"]'))
    )
    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '/html/body/main/div/div/div/form/div[4]/div[2]/button'))
    )
    
    username_input.send_keys(USERNAME)
    password_input.send_keys(PASSWORD)
    login_button.click()
    print("login_button clicked ok")

    # Wait for login to complete
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "button-menu-home-hover"))
    )
    print("Login successful")

    # Get User-Agent from Selenium
    user_agent = driver.execute_script("return navigator.userAgent;")
    print("User-Agent from Selenium:", user_agent)

    # Create a requests session and copy cookies & headers
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        # 如果需要Referer或其他header也在此加入
        "Referer": ARCHIVE_URL
    })

    # 將Selenium的cookies塞入requests session
    for cookie in driver.get_cookies():
        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

    # Navigate to archive page
    driver.get(ARCHIVE_URL)
    print("Accessed archive page")

    while True:
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/main/div/div/div[2]/div/div[2]/div/div[1]/div/div/div[1]/table/thead/tr[2]/th[2]"))
            )
            if element.text == "October 2024":
                print("Found the month October 2024!")
                os.makedirs(os.path.join(DOWNLOAD_FOLDER, str(element.text)), exist_ok=True)
                break
            else:
                button_nxtmth = driver.find_element(By.XPATH, "/html/body/main/div/div/div[2]/div/div[2]/div/div[1]/div/div/div[1]/table/thead/tr[2]/th[1]")
                button_nxtmth.click()
        except TimeoutException:
            print("Timeout waiting for the element, check the XPath.")
            break

    while not check_month_and_year("December 20255"):
        element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "/html/body/main/div/div/div[2]/div/div[2]/div/div[1]/div/div/div[1]/table/thead/tr[2]/th[2]"))
                )
        os.makedirs(os.path.join(DOWNLOAD_FOLDER, str(element.text)), exist_ok=True)

        element_this_round = str(element.text)
        print("***element_this_round***is ", element_this_round)

        # 遍歷日期表格
        for k in range(1, 7):
            for y in range(1,8):
                button_xpath = f"/html/body/main/div/div/div[2]/div/div[2]/div/div[1]/div/div/div[1]/table/tbody/tr[{k}]/td[{y}]"
                print(f"Date index [{k},{y}]")
                try:
                    button = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, button_xpath))
                    )
                    class_attribute = button.get_attribute("class")
                    if "day" in class_attribute and "new day" not in class_attribute and "old day" not in class_attribute and "disabled day" not in class_attribute:
                        button.click()
                        time.sleep(2)

                        date_selector = wait.until(EC.presence_of_element_located((By.ID, "datepicker")))
                        available_days = date_selector.find_elements(By.XPATH, ".//td[contains(@class, 'day')]")

                        table_rows = driver.find_elements(By.XPATH, "/html/body/main/div/div/div[2]/div/div[2]/div/div[2]/div/div[2]/div/table/tbody/tr")
                        for x in range(1, len(table_rows) + 1):
                            try:
                                print("====start======================================")
                                button_dwn = driver.find_element(By.XPATH, f"/html/body/main/div/div/div[2]/div/div[2]/div/div[2]/div/div[2]/div/table/tbody/tr[{x}]/td[2]/a")
                                old_url = driver.current_url
                                mp3_url = button_dwn.get_attribute("href")
                                print(f"Requesting: {mp3_url}")

                                driver.execute_script("arguments[0].click();", button_dwn)
                                print("Button clicked via JavaScript")

                                WebDriverWait(driver, 10).until(lambda d: d.current_url != old_url)
                                new_url = driver.current_url
                                print(f"Redirected to new URL: {new_url}")

                                # 使用session下載確保cookies與user-agent一致
                                response = session.get(mp3_url, stream=True, allow_redirects=True)

                                final_url = response.url
                                print(f"Final redirected URL: {final_url}")
                                print(f"Status Code: {response.status_code}")
                                print(f"Headers: {response.headers}")
                                print("====done======================================")

                                if response.status_code == 200 and "audio/mpeg" in response.headers.get("Content-Type", ""):
                                    file_name = os.path.join(DOWNLOAD_FOLDER, os.path.basename(final_url))
                                    print(f"Downloading to: {file_name}")
                                    with open(file_name, "wb") as file:
                                        for chunk in response.iter_content(chunk_size=1024):
                                            if chunk:
                                                file.write(chunk)
                                    print(f"Downloaded: {file_name}")

                                    time.sleep(2)

                                    # After download, navigate back to archive URL
                                    driver.get(ARCHIVE_URL)
                                    print("Returned to archive page")
                                    time.sleep(2)  # Wait for the page to reload

                                    navigate_to_month_and_prepare_directory(driver, element_this_round)

                                    button_xpath = f"/html/body/main/div/div/div[2]/div/div[2]/div/div[1]/div/div/div[1]/table/tbody/tr[{k}]/td[{y}]"
                                    button = WebDriverWait(driver, 10).until(
                                            EC.presence_of_element_located((By.XPATH, button_xpath))
                                    )
                                    button.click()
                                    time.sleep(2)  # Wait for the page to reload
                                else:
                                    print(f"Failed to download {mp3_url}. Content-Type: {response.headers.get('Content-Type')}, Status code: {response.status_code}")
                            except Exception as e:
                                print(f"Error processing file {x}: {e}")
                except Exception as ee:
                    print(f"No clickable 'day' button at position [{k},{y}]: {str(ee)}")

        print("Download files for the month")
        click_button("/html/body/main/div/div/div[2]/div/div[2]/div/div[1]/div/div/div[1]/table/thead/tr[2]/th[3]")

    print("All dates processed successfully")

finally:
    time.sleep(10)
    driver.quit()

from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

def page_has_loaded(driver):
    page_state = driver.execute_script('return document.readyState;')
    return page_state == 'complete'

def login(driver, email=None, password=None):
    if email is None or password is None:
        print("*** ERROR: missing email or password")
        exit(0)

    # Request login page
    driver.get("https://www.linkedin.com/login")

    # Wait for the LinkedIn login page to load completely
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "username")))

    # Find email input
    email_elem = driver.find_element_by_id("username")
    email_elem.send_keys(email)

    # Find password input
    password_elem = driver.find_element_by_id("password")
    password_elem.send_keys(password)
    driver.find_element_by_tag_name("button").click()

    # Wait for the LinkedIn user dashboard to load completely
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "profile-nav-item")))
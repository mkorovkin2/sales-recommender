import pandas as pd
import numpy as np

from linkedin_sales_recommender import actions_util
from selenium import webdriver

# Example data lists provided below
example_positive_list = [
    "brpitts",
    "bradstantonatx",
    "stephencwinslow",
    "emorganboston",
    "benjamin-nye-1766673"
]

example_negative_list = [
    "takashirono",
    "samikshro",
    "akashkapur1",
    "pedrovirguezbogotacolombia",
    "koladeaderele"
]

example_individual_list = example_positive_list + example_negative_list

example_company_list = [
    'jbcholdings-llc',
    'soprasteria',
    'itransition',
    'thesoftwareguild',
    'programmer-resources-intl-inc'
]

# Set up the Selenium driver and global variables
driver = webdriver.Chrome("./chromedriver")

information_list = []
information_set = set()

login_email = ""
login_password = ""

# Need to log into LinkedIn in order to run this code
actions_util.login(driver, email=login_email, password=login_password)

# Base strings for either type of request
base_string_company = "https://www.linkedin.com/company/"
base_string_individual = "https://www.linkedin.com/in/"

def _clean_text(item_text):
    """Cleans the text of a LinkedIn user's profile to extract key informataion

    Parameters
    ----------
    item_text : str
        Text to clean, presumably a summary of an individual's profile

    Returns
    -------
    list
        The list contains [company, location, description of position, duration of position, and position title]

    """

    # Define all variables
    company = ""
    location = ""
    description = ""
    duration = ""

    # Split text at line breaks
    split_text = item_text.split("\n")

    # Extract position label
    position = "" if split_text[0] == "Title" else split_text[0]

    # Extract other attributes of text if they exist
    for i in range(len(split_text)):
        if split_text[i] == "Company Name":
            company = split_text[i + 1] if len(split_text) > i + 1 else ""

        if split_text[i] == "Title":
            position = split_text[i + 1] if len(split_text) > i + 1 else ""

        if split_text[i] == "Employment Duration":
            duration = split_text[i + 1] if len(split_text) > i + 1 else ""

        if split_text[i] == "Location":
            location = split_text[i + 1] if len(split_text) > i + 1 else ""

            # Get description of position if it exists
            description = split_text[i + 2:] if len(split_text) > i + 2 else ""
            description_string = ""
            for d in description:
                description_string += "\n" + d

            # Update description string
            description = description_string[1:]

    return [company, location, description, duration, position]

def get_list_of_company_urls(pages=50, search_query="software"):
    """Gets list of company LinkedIn URLs based on a search query

    Parameters
    ----------
    pages : int
        Number of search pages to scrape companies from
    search_query : str
        Query for the company search

    Returns
    -------
    list
        List of company URLs

    """

    # Iterate through pages
    for page in range(pages):
        try:

            # Make a link string and request it
            link = f"https://www.linkedin.com/search/results/companies/?keywords=" + search_query.replace(" ", "%20")\
                   + "&origin=GLOBAL_SEARCH_HEADER&page=" + str(page + 1)

            driver.get(link)

            # Get the full resulting page with a list of companies
            full_result = driver.find_elements_by_xpath('//a[@data-control-name = "search_srp_result"]')
            for element in full_result:

                # Identify the company URL in each company
                full_link = element.get_attribute('href')
                gsm = full_link.replace("www.linkedin.com/company/", "").replace("/", "").split(":")[-1]

                # Add information to stored list
                information_set.add(gsm)

        except Exception as e:
            print("*** ERROR - current list of links:", list(information_set))
            driver.quit()
            exit(0)

    # Close the Chrome window
    driver.quit()

    return list(information_set)

def get_list_of_individual_urls(pages=50, search_query="sales"):
    """Gets list of individual LinkedIn URLs based on a search query

    Parameters
    ----------
    pages : int
        Number of search pages to scrape individuals from
    search_query : str
        Query for the individual search

    Returns
    -------
    list
        List of individual profile URLs

    """

    # Iterate through pgaes
    for page in range(pages):
        try:

            # Construct a link for the request and request it
            link = "https://www.linkedin.com/search/results/people/?keywords=" + search_query.replace(" ", "%20")\
                   + "&origin=GLOBAL_SEARCH_HEADER&page=" + str(page + 1)

            driver.get(link)

            # Get the resulting list of individuals
            full_result = driver.find_elements_by_xpath('//a[@data-control-name = "search_srp_result"]')
            for element in full_result:

                # Find the link attribute
                full_link = element.get_attribute('href')
                gsm = full_link.replace("www.linkedin.com/in/", "").replace("/", "").split(":")[-1]

                # Log the link attribute
                information_set.add(gsm)

        except Exception as e:
            print("*** ERROR - current list of links:", list(information_set))
            driver.quit()
            exit(0)

    # Close the driver
    driver.quit()

    return list(information_set)

def scrape_individuals(linkedin_individual_url_list, output_file_name, backup_frequency=50, pre_label="No"):
    """Scrapes descriptions of individuals' work histories from individual URL list and generates a resulting CSV file

    Parameters
    ----------
    linkedin_individual_url_list : list
        List of individuals' LinkedIn profile URLs
    output_file_name : str
        CSV output file name (without extension)
    backup_frequency : int
        Number of user profiles scraped between each automated backup of the dataframe
    pre_label : str
        Pre-generated label for each individual ("No" = negative classification, "Yes" = positive classification)

    Returns
    -------
    Nothing

    """

    iteration_counter = 0

    # For each link in the link list, scrape the user's information and organize it
    for link_string in linkedin_individual_url_list:
        try:
            # Request URL
            driver.get(base_string_individual + link_string)

            # Get experience
            experience = driver.find_elements_by_xpath('//section[@id = "experience-section"]/ul//li')

            # Iterate through experience and treat each experience as a separate item
            for item in experience:

                # Log the individual's attributes with a pre-generated label
                ret_array = _clean_text(item.text)
                net = [link_string] + ret_array + [pre_label]
                if net[1] == "":
                    net[1] = information_list[-1][1]
                if net[5] == "Title":
                    net[5] = information_list[-1][5]

                # Add experience into list
                information_list.append(net)

            # Update counter variable
            iteration_counter += 1

            # Backup the dataframe occasionally
            if iteration_counter % backup_frequency == 0 and iteration_counter > 1:
                print("Generating dataframe backup...")

                dfz = pd.DataFrame(columns=[
                    "name", "company", "location", "description", "duration", "position", "rec"
                ], data=np.array(information_list))

                # Create CSV backup file
                dfz.to_csv("standard_backup_individual_" + output_file_name + "_iteration" + str(iteration_counter) + ".csv")

        except Exception as e:
            print("*** Exception")
            print(e)

            # If an exception occurs, create a dataframe backup file
            dfz = pd.DataFrame(columns=["name", "company", "location", "description", "duration", "position", "rec"],
                               data=np.array(information_list))
            dfz.to_csv("error_backup_individual_" + output_file_name + "_iteration" + str(iteration_counter) + ".csv")

            continue

    # Build the final dataframe
    df = pd.DataFrame(columns=[
        "name", "company", "location", "description", "duration", "position", "rec"
    ], data=np.array(information_list))

    # Create CSV file from dataframe
    df.to_csv(output_file_name + "_final.csv")

    # Quit the Chrome driver
    driver.quit()

def scrape_companies(linkedin_company_url_list, output_file_name, pre_label="No"):
    """Scrapes descriptions of companies from company URL list and generates a resulting CSV file

    Parameters
    ----------
    linkedin_company_url_list : list
        List of company LinkedIn page URLs
    output_file_name : str
        CSV output file name (without extension)
    pre_label : str
        Pre-generated label for each company ("No" = negative classification, "Yes" = positive classification)

    Returns
    -------
    Nothing

    """

    # Iterate through each company link and organize its results
    for link_string in linkedin_company_url_list:
        try:
            # Request URL
            driver.get(base_string_company + link_string + "/about/")

            # Get the company's name
            name = driver.find_element_by_xpath(
                '//h1[@class = "org-top-card-summary__title t-24 t-black truncate"]/span')

            # Get the company's description
            description = driver.find_elements_by_xpath(
                '//p[@class = "break-words white-space-pre-wrap mb5 t-14 t-black--light t-normal"]')

            # Log the company's name and description with a pre-generated label
            for item in description:
                information_list.append([link_string, name.text.strip(), item.text.replace("\n", " "), pre_label])

        except Exception as e:
            print("*** Exception")
            print(e)

            # If an exception occurs, create a dataframe backup file
            df_temp = pd.DataFrame(columns=["company", "name", "description", "label"], data=np.array(information_list))
            df_temp.to_csv("error_backup_company_" + output_file_name + ".csv")

    # Build the final dataframe
    df = pd.DataFrame(columns=["company", "name", "description", "label"], data=np.array(information_list))

    # Create CSV file from dataframe
    df.to_csv(output_file_name + "_final.csv")

    # Quit the Chrome Driver
    driver.quit()
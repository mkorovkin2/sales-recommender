# LinkedIn Sales Recommender

<i>Developed by Michael Korovkin </i>

## Usage
This package is used as proof-of-concept for using machine learning in the task of
determining a given person's quality of fit for a sales role in software consulting.

Files can be run through command line.

## Package Description
* ```util.py```: provides utility for LinkedIn login and waiting for dashboard to load
* ```scraper.py```: provides resources to scrape lists of individuals and companies,
in addition to to scraping attributes of companies and individuals' work experiences
* ```predictor.py```: contains classification modeling and performance analysis
## Setup
<i>Note that the ```chromedriver``` binary is in the same directory as ```scraper.py```</i>

### Getting Data
##### Process to obtain a list of individuals from a LinkedIn query
In ```scraper.py```, run the method called ```get_list_of_individual_urls```.
This method will output a list of individuals' profile URLs, up to a specified page in
LinkedIn's search functionality given a search query.

##### Process to obtain a list of companies from a LinkedIn query
In ```scraper.py```, run the method called ```get_list_of_company_urls```.
This method will output a list of company page URLs, up to a specified page in
LinkedIn's search functionality given a search query.

##### Get work experience histories for a list of individuals
Given a list of individuals' profile URLs, run the method ```scrape_individuals``` in
```scraper.py```. This will create a CSV file with the individuals' work history, including
attributes:
* Profile URL
* Company name
* Work location
* Position description
* Duration of work
* Position title

##### Get work experience histories for a list of individuals
Given a list of individuals' profile URLs, run the method ```scrape_companies``` in
```scraper.py```. This will create a CSV file with the companies descriptions, which
include attributes:
* Page URL
* Company description
* Company name

### Running Predictions
##### Analyzing performance of software sales role prediction
Given CSV files of company descriptions and individuals' previous work histories, in
the same directory as ```predictor.py```, run ```predictor.py```. The predictor will
attempt to find the best prediction boundary using multiple samples, and will output
statistics on its performance and ability to predict the correct fit for individuals.

Decision boundaries are used to determine positive/negative classifications and provide
more flexibility to classifying sales people properly. Boundaries generally range from
0.5 to 0.7 (with scores lower than the boundary corresponding to negative classifications
and scores higher than the boundary corresponding to positive classifications). Boundaries
are explored at strides of 0.01, starting at 0.50, up to the user's input (default: 0.70). 

The code can be explored and tested with the example CSV files provided in the package.
These files are ```example_company_dataframe.csv``` and ```example_position_dataframe.csv```.

##### Running the analysis through command line
Parameters
* -w, --workhistory: work history dataframe location
* -c, --company: company dataframe location
* -s, --samples: number of samples per boundary check
* -t, --topbound: maximum boundary (calculated as (maximum boundary - 0.5) * 100)

Command: ```python predictor.py -w -c -s -t```

---

Example command usage:
* w: default
* c: default
* s: 100 samples
* t: stopping at a top boundary of 0.65

Command: ```python predictor.py -s 100 -t 15```

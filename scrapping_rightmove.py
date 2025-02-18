
import datetime
from lxml import html, etree
import numpy as np
import pandas as pd
import requests
from lxml.cssselect import CSSSelector
import time




class RightmoveData:
    """The `RightmoveData` webscraper collects structured data on properties
    returned by a search performed on www.rightmove.co.uk

    An instance of the class provides attributes to access data from the search
    results, the most useful being `get_results`, which returns all results as a
    Pandas DataFrame object.

    The query to rightmove can be renewed by calling the `refresh_data` method.
    """
    def __init__(self, url: str, get_floorplans: bool = False):
        """Initialize the scraper with a URL from the results of a property
        search performed on www.rightmove.co.uk.

        Args:
            url (str): full HTML link to a page of rightmove search results.
            get_floorplans (bool): optionally scrape links to the individual
                floor plan images for each listing (be warned this drastically
                increases runtime so is False by default).
        """
        self._status_code, self._first_page = self._request(url)
        self._url = url
        self._validate_url()
        self._results = self._get_results(get_floorplans=get_floorplans)

    @staticmethod
    def _request(url: str):
        r = requests.get(url)
        return r.status_code, r.content

    def refresh_data(self, url: str = None, get_floorplans: bool = False):
        """Make a fresh GET request for the rightmove data.

        Args:
            url (str): optionally pass a new HTML link to a page of rightmove
                search results (else defaults to the current `url` attribute).
            get_floorplans (bool): optionally scrape links to the individual
                flooplan images for each listing (this drastically increases
                runtime so is False by default).
        """
        url = self.url if not url else url
        self._status_code, self._first_page = self._request(url)
        self._url = url
        self._validate_url()
        self._results = self._get_results(get_floorplans=get_floorplans)

    def _validate_url(self):
        """Basic validation that the URL at least starts in the right format and
        returns status code 200."""
        real_url = "{}://www.rightmove.co.uk/{}/find.html?"
        protocols = ["http", "https"]
        types = ["property-to-rent", "property-for-sale", "new-homes-for-sale"]
        urls = [real_url.format(p, t) for p in protocols for t in types]
        conditions = [self.url.startswith(u) for u in urls]
        conditions.append(self._status_code == 200)
        if not any(conditions):
            raise ValueError(f"Invalid rightmove search URL:\n\n\t{self.url}")

    @property
    def url(self):
        return self._url

    @property
    def get_results(self):
        """Pandas DataFrame of all results returned by the search."""
        return self._results

    @property
    def results_count(self):
        """Total number of results returned by `get_results`. Note that the
        rightmove website may state a much higher number of results; this is
        because they artificially restrict the number of results pages that can
        be accessed to 42."""
        return len(self.get_results)

    @property
    def average_price(self):
        """Average price of all results returned by `get_results` (ignoring
        results which don't list a price)."""
        total = self.get_results["price"].dropna().sum()
        return total / self.results_count

    def summary(self, by: str = None):
        """DataFrame summarising results by mean price and count. Defaults to
        grouping by `number_bedrooms` (residential) or `type` (commercial), but
        accepts any column name from `get_results` as a grouper.

        Args:
            by (str): valid column name from `get_results` DataFrame attribute.
        """
        if not by:
            by = "type" if "commercial" in self.rent_or_sale else "number_bedrooms"
        assert by in self.get_results.columns, f"Column not found in `get_results`: {by}"
        df = self.get_results.dropna(axis=0, subset=["price"])
        groupers = {"price": ["count", "mean"]}
        df = df.groupby(df[by]).agg(groupers)
        df.columns = df.columns.get_level_values(1)
        df.reset_index(inplace=True)
        if "number_bedrooms" in df.columns:
            df["number_bedrooms"] = df["number_bedrooms"].astype(int)
            df.sort_values(by=["number_bedrooms"], inplace=True)
        else:
            df.sort_values(by=["count"], inplace=True, ascending=False)
        return df.reset_index(drop=True)

    @property
    def rent_or_sale(self):
        """String specifying if the search is for properties for rent or sale.
        Required because Xpaths are different for the target elements."""
        if "/property-for-sale/" in self.url or "/new-homes-for-sale/" in self.url:
            return "sale"
        elif "/property-to-rent/" in self.url:
            return "rent"
        elif "/commercial-property-for-sale/" in self.url:
            return "sale-commercial"
        elif "/commercial-property-to-let/" in self.url:
            return "rent-commercial"
        else:
            raise ValueError(f"Invalid rightmove URL:\n\n\t{self.url}")

    @property
    def results_count_display(self):
        """Returns an integer of the total number of listings as displayed on
        the first page of results. Note that not all listings are available to
        scrape because rightmove limits the number of accessible pages."""
        tree = html.fromstring(self._first_page)
        xpath = """//span[@class="searchHeader-resultCount"]/text()"""
        return int(tree.xpath(xpath)[0].replace(",", ""))

    @property
    def page_count(self):
        """Returns the number of result pages returned by the search URL. There
        are 24 results per page. Note that the website limits results to a
        maximum of 42 accessible pages."""
        page_count = self.results_count_display // 24
        if self.results_count_display % 24 > 0:
            page_count += 1
        # Rightmove will return a maximum of 42 results pages, hence:
        if page_count > 42:
            page_count = 42
        return page_count
    
    def _get_page(self, request_content: str, get_floorplans: bool = False):
        """Method to scrape data from a single page of search results. Used
        iteratively by the `get_results` method to scrape data from every page
        returned by the search."""
        # Process the html:
        tree = html.fromstring(request_content)
        # Print tree on the first run
        #if not hasattr(self, '_first_run'):
        html_string = etree.tostring(tree, pretty_print=True, encoding='unicode')
            #self._first_run = False
            # Save the HTML string to a file
            #with open('output.html', 'w', encoding='utf-8') as file:
                #file.write(html_string)
        # Set xpath for price:
        if "rent" in self.rent_or_sale:
            xp_prices = """//span[@class="propertyCard-priceValue"]/text()"""
        elif "sale" in self.rent_or_sale:
            xp_prices = """//div[@class="propertyCard-priceValue"]/text()"""
        else:
            raise ValueError("Invalid URL format.")
        
        # Set xpaths for listing title, property address, URL, and agent URL:
        xp_titles = """//div[@class="propertyCard-details"]\
        //a[@class="propertyCard-link"]\
        //h2[@class="propertyCard-title"]/text()"""
        xp_addresses = """//address[@class="propertyCard-address"]//span/text()"""
        xp_weblinks = """//div[@class="propertyCard-details"]//a[@class="propertyCard-link"]/@href"""
        xp_agent_urls = """//div[@class="propertyCard-contactsItem"]\
        //div[@class="propertyCard-branchLogo"]\
        //a[@class="propertyCard-branchLogo-link"]/@href"""
        
        
        xp_dates_added = '/html/body/div[1]/div[2]/div[1]/div[2]/div[5]/div[1]/div[2]/div/div[1]/div[4]/div[2]/div[3]/span[1]'
        
        # Use regular expression to extract the JSON data
        json_match = re.search(r'window\.jsonModel\s*=\s*({.*?})\s*(?=<)', html_string, re.DOTALL | re.MULTILINE)


        if json_match:
            json_data = json_match.group(1)

            # Parse the JSON data using json5
            json_load_data = json.loads(json_data)

            # Extract desired data for each property
            properties_data = []
            for property_data in json_load_data['properties']:
                property_info = {
                    'id': property_data.get('id'),
                    'bedrooms': property_data.get('bedrooms'),
                    'bathrooms': property_data.get('bathrooms'),
                    'numberOfImages': property_data.get('numberOfImages'),
                    'numberOfFloorplans': property_data.get('numberOfFloorplans'),
                    'numberOfVirtualTours': property_data.get('numberOfVirtualTours'),
                    'latitude': property_data.get('location', {}).get('latitude'),
                    'longitude': property_data.get('location', {}).get('longitude'),
                    'propertySubType': property_data.get('propertySubType'),
                    'listingUpdateReason': property_data.get('listingUpdate', {}).get('listingUpdateReason'),
                    'listingUpdateDate': property_data.get('listingUpdate', {}).get('listingUpdateDate'),
                    'premiumListing': property_data.get('premiumListing'),
                    'featuredProperty': property_data.get('featuredProperty'),
                    'price_json': property_data.get('price', {}).get('amount'),
                    'priceFrequency': property_data.get('price', {}).get('frequency'),
                    'priceCurrencyCode': property_data.get('price', {}).get('currencyCode'),
                    'displayPrice': property_data.get('price', {}).get('displayPrices', [{}])[0].get('displayPrice'),
                    'displayPriceQualifier': property_data.get('price', {}).get('displayPrices', [{}])[0].get('displayPriceQualifier'),
                    'branchDisplayName': property_data.get('customer', {}).get('branchDisplayName'),
                    'firstVisibleDate': property_data.get('firstVisibleDate')
                }
                properties_data.append(property_info)
        # Access the desired information from the parsed data
        
        # Create data lists from xpaths:
        price_pcm = tree.xpath(xp_prices)
        titles = tree.xpath(xp_titles)
        addresses = tree.xpath(xp_addresses)
        base = "http://www.rightmove.co.uk"
        weblinks = [f"{base}{tree.xpath(xp_weblinks)[w]}" for w in range(len(tree.xpath(xp_weblinks)))]
        agent_urls = [f"{base}{tree.xpath(xp_agent_urls)[a]}" for a in range(len(tree.xpath(xp_agent_urls)))]
        #bathroom_numbers = tree.cssselect(xp_bathroom_numbers)
        #agent_names = tree.xpath(xp_agent_names)
        #photo_counts = tree.xpath(xp_photo_counts)
        #dates_added = tree.xpath(xp_dates_added)
                
        dates_added = [date.text for date in tree.xpath(xp_dates_added)]
        

        # Optionally get floorplan links from property urls (longer runtime):
        #print(dates_added)
        
        floorplan_urls = list() if get_floorplans else np.nan
        if get_floorplans:
            for weblink in weblinks:
                status_code, content = self._request(weblink)
                if status_code != 200:
                    continue
                tree = html.fromstring(content)
                xp_floorplan_url = """//*[@id="floorplanTabs"]/div[2]/div[2]/img/@src"""
                floorplan_url = tree.xpath(xp_floorplan_url)
                if floorplan_url:
                    floorplan_urls.append(floorplan_url[0])
                else:
                    floorplan_urls.append(np.nan)

        # Create a DataFrame from the extracted data
        df_properties = pd.DataFrame(properties_data)
        
        # Merge the extracted data with the existing data
        data = [price_pcm, titles, addresses, weblinks, agent_urls, dates_added]
        data = data + [floorplan_urls] if get_floorplans else data
        temp_df = pd.DataFrame(data)
        temp_df = temp_df.transpose()
        columns = ["price", "type", "address", "url", "agent_url",  "date_added"]
        columns = columns + ["floorplan_url"] if get_floorplans else columns
        temp_df.columns = columns
        try:
        # Merge the extracted data DataFrame with the existing DataFrame
            merged_df = pd.concat([temp_df, df_properties], axis=1)
        except:
            pass
        # Drop empty rows which come from placeholders in the html:
        merged_df = merged_df[merged_df["address"].notnull()]
        #print(merged_df)
        return merged_df

    def _get_results(self, get_floorplans: bool = False):
        """Build a Pandas DataFrame with all results returned by the search."""
        results = self._get_page(self._first_page, get_floorplans=get_floorplans)

        # Reset the index of the initial results DataFrame:
        

        # Iterate through all pages scraping results:
        for p in range(1, self.page_count + 1, 1):
            # Create the URL of the specific results page:
            p_url = f"{str(self.url)}&index={p * 24}"

            # Make the request:
            status_code, content = self._request(p_url)

            # Requests to scrape lots of pages eventually get status 400, so:
            if status_code != 200:
                break

            # Create a temporary DataFrame of page results:
            temp_df = self._get_page(content, get_floorplans=get_floorplans)

            # Reset the index of the temporary DataFrame:
            


            try:
            # Concatenate the temporary DataFrame with the full DataFrame:
                results = pd.concat([results, temp_df], ignore_index=True)
            except:
                print('merge error')
                pass
        return self._clean_results(results)

    @staticmethod
    def _clean_results(results: pd.DataFrame):

        # Reset the index:
        results.reset_index(inplace=True, drop=True)

        # Convert price column to numeric type:
        results["price"].replace(regex=True, inplace=True, to_replace=r"\D", value=r"")
        
        results["price"] = pd.to_numeric(results["price"])

        # Extract short postcode area to a separate column:
        pat = r"\b([A-Za-z][A-Za-z]?[0-9][0-9]?[A-Za-z]?)\b"
        results["postcode"] = results["address"].astype(str).str.extract(pat, expand=True)[0]

        # Extract full postcode to a separate column:
        pat = r"([A-Za-z][A-Za-z]?[0-9][0-9]?[A-Za-z]?[0-9]?\s[0-9]?[A-Za-z][A-Za-z])"
        results["full_postcode"] = results["address"].astype(str).str.extract(pat, expand=True)[0]

        # Extract number of bedrooms from `type` to a separate column:
        pat = r"\b([\d][\d]?)\b"
        results["number_bedrooms"] = results["type"].astype(str).str.extract(pat, expand=True)[0]
        results.loc[results["type"].str.contains("studio", case=False), "number_bedrooms"] = 0
        results["number_bedrooms"] = pd.to_numeric(results["number_bedrooms"])

        # Clean up annoying white spaces and newlines in `type` column:
        results["type"] = results["type"].str.strip("\n").str.strip()

        # Add column with datetime when the search was run (i.e. now):
        now = datetime.now()
        results["search_date"] = now
        
        return results
    
import ast
import re

from datetime import datetime
from lxml import html
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

# Env
address_pattern = r'([\s\S]+?)([A-Za-z][A-Za-z]?[0-9][0-9]?[A-Za-z]?[0-9]?\s[0-9]?[A-Za-z][A-Za-z])'
outwardcode_pattern = r'([A-Za-z][A-Za-z]?[0-9][0-9]?[A-Za-z]?[0-9]?)'

# Helpers
def extract_price(series):
    prices = []
    for entry in series:
        prices.append(int(entry[0]['displayPrice'].strip('£').replace(',', '')))
    return prices


def extract_date(series):
    dates = []
    for entry in series:
        dates.append(datetime.strptime(entry[0]['dateSold'], '%d %b %Y'))
    return dates


def extract_tenure(series):
    tenures = []
    for entry in series:
        tenures.append(entry[0]['tenure'])
    return tenures


def extract_coords(series, lat=False):
    coords = []
    if lat:
        for entry in series:
            coords.append(entry['lat'])
    else:
        for entry in series:
            coords.append(entry['lng'])
    return coords

class SoldProperties:

    def __init__(self, url: str, get_floorplans: bool = False):
        """Initialize the scraper with a URL from the results of a property
        search performed on www.rightmove.co.uk.

        Args:
            url (str): full HTML link to a page of rightmove search results.
            get_floorplans (bool): optionally scrape links to the individual
                floor plan images for each listing (be warned this drastically
                increases runtime so is False by default).
        """
        self._status_code, self._first_page = self._request(url)
        self._url = url
        self._validate_url()
        self._results = self._get_results()

    @staticmethod
    def _request(url: str):
        r = requests.get(url)
        return r.status_code, r.content

    def refresh_data(self, url: str = None, get_floorplans: bool = False):
        """Make a fresh GET request for the rightmove data.

        Args:
            url (str): optionally pass a new HTML link to a page of rightmove
                search results (else defaults to the current `url` attribute).
            get_floorplans (bool): optionally scrape links to the individual
                flooplan images for each listing (this drastically increases
                runtime so is False by default).
        """
        url = self.url if not url else url
        self._status_code, self._first_page = self._request(url)
        self._url = url
        self._validate_url()
        self._results = self._get_results()

    def _validate_url(self):
        """Basic validation that the URL at least starts in the right format and
        returns status code 200."""
        real_url = "{}://www.rightmove.co.uk/{}/find.html?"
        protocols = ["http", "https"]
        types = ["property-to-rent", "property-for-sale", "new-homes-for-sale"]
        urls = [real_url.format(p, t) for p in protocols for t in types]
        conditions = [self.url.startswith(u) for u in urls]
        conditions.append(self._status_code == 200)
        if not any(conditions):
            raise ValueError(f"Invalid rightmove search URL:\n\n\t{self.url}")

    @property
    def url(self):
        return self._url

    @property
    def table(self):
        return self._results

    def _parse_page_data_of_interest(self, request_content: str):
        """Method to scrape data from a single page of search results. Used
        iteratively by the `get_results` method to scrape data from every page
        returned by the search."""
        soup = BeautifulSoup(request_content, features='lxml')

        start = 'window.__PRELOADED_STATE__ = '
        tags = soup.find_all(
            lambda tag: tag.name == 'script' and start in tag.get_text())
        if not tags:
            raise ValueError('Could not extract data from current page!')
        if len(tags) > 1:
            raise ValueError('Inconsistent data in current page!')

        json_str = tags[0].get_text()[len(start):]
        json_obj = json.loads(json_str)

        return json_obj

    def _get_properties_list(self, json_obj):
        return json_obj['results']['properties']

    def _get_results(self):
        """Build a Pandas DataFrame with all results returned by the search."""
        print('Scraping page {}'.format(1))
        print('- Parsing data from page {}'.format(1))
        try:
            page_data = self._parse_page_data_of_interest(self._first_page)
            properties = self._get_properties_list(page_data)
        except ValueError:
            print('Failed to get property data from page {}'.format(1))

        final_results = properties

        current = page_data['pagination']['current']
        last = page_data['pagination']['last']
        if current == last:
            return

        # Scrape each page
        for page in range(current + 1, last):
            print('Scraping page {}'.format(page))

            # Create the URL of the specific results page:
            p_url = f"{str(self.url)}&page={page}"

            # Make the request:
            print('- Downloading data from page {}'.format(page))
            status_code, page_content = self._request(p_url)

            # Requests to scrape lots of pages eventually get status 400, so:
            if status_code != 200:
                print('Failed to access page {}'.format(page))
                continue

            # Create a temporary DataFrame of page results:
            print('- Parsing data from page {}'.format(page))
            try:
                page_data = self._parse_page_data_of_interest(page_content)
                properties = self._get_properties_list(page_data)
            except ValueError:
                print('Failed to get property data from page {}'.format(page))

            # Append the list or properties.
            final_results += properties

        # Transform the final results into a table.
        property_data_frame = pd.DataFrame.from_records(final_results)

        def process_data(rawdf):
            df = rawdf.copy()
        
            address = df['address'].str.extract(address_pattern, expand=True).to_numpy()
            outwardcodes = df['address'].str.extract(outwardcode_pattern, expand=True).to_numpy()
            
            df = (df.drop(['address', 'images', 'hasFloorPlan', 'detailUrl'], axis=1)
                    .assign(address=address[:, 0])
                    .assign(postcode=address[:, 1])
                    .assign(outwardcode=outwardcodes[:, 0])
                    #.assign(transactions=df.transactions.apply(ast.literal_eval))
                    #.assign(location=df.location.apply(ast.literal_eval))
                    .assign(last_price=lambda x: extract_price(x.transactions))
                    .assign(sale_date=lambda x: extract_date(x.transactions))
                    .assign(tenure=lambda x: extract_tenure(x.transactions))
                    .assign(lat=lambda x: extract_coords(x.location, lat=True))
                    .assign(lng=lambda x: extract_coords(x.location))
                    .drop(['transactions', 'location'], axis=1)
            )
            return df
     
        #return process_data(property_data_frame)

        return property_data_frame

    @property
    def processed_data(self):
        df = self._results
    
        address = df['address'].str.extract(address_pattern, expand=True).to_numpy()
        outwardcodes = df['address'].str.extract(outwardcode_pattern, expand=True).to_numpy()
        
        df = (df.drop(['address', 'images', 'hasFloorPlan', 'detailUrl'], axis=1)
                .assign(address=address[:, 0])
                .assign(postcode=address[:, 1])
                .assign(outwardcode=outwardcodes[:, 0])
                #.assign(transactions=df.transactions.apply(ast.literal_eval))
                #.assign(location=df.location.apply(ast.literal_eval))
                .assign(last_price=lambda x: extract_price(x.transactions))
                .assign(sale_date=lambda x: extract_date(x.transactions))
                .assign(tenure=lambda x: extract_tenure(x.transactions))
                .assign(lat=lambda x: extract_coords(x.location, lat=True))
                .assign(lng=lambda x: extract_coords(x.location))
                .drop(['transactions', 'location'], axis=1)
                .reindex(columns=['last_price', 
                                'sale_date', 
                                'propertyType',
                                'bedrooms',
                                'bathrooms', 
                                'tenure', 
                                'address', 
                                'postcode', 
                                'outwardcode', 
                                'lat', 
                                'lng'])
        )
        return df
   
url = 'https://www.rightmove.co.uk/property-for-sale/find.html?searchType=SALE&locationIdentifier=REGION%5E1477&insId=1&radius=0.0&minPrice=&maxPrice=&minBedrooms=&maxBedrooms=&displayPropertyType=&maxDaysSinceAdded=&_includeSSTC=on&sortByPriceDescending=&primaryDisplayPropertyType=&secondaryDisplayPropertyType=&oldDisplayPropertyType=&oldPrimaryDisplayPropertyType=&newHome=&auction=false'
suffolkurl = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk1 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&maxBedrooms=1&minBedrooms=0&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk2 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&maxBedrooms=2&minBedrooms=2&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk3 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&maxBedrooms=2&minBedrooms=2&sortType=1&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk4 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&maxBedrooms=3&minBedrooms=3&maxPrice=280000&sortType=1&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk5 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&maxBedrooms=3&minBedrooms=3&maxPrice=350000&minPrice=280000&sortType=1&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk6 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&maxBedrooms=3&minBedrooms=3&maxPrice=700000&minPrice=350000&sortType=1&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk7 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&maxBedrooms=3&minBedrooms=3&minPrice=700000&sortType=1&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk8 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&maxBedrooms=4&minBedrooms=4&sortType=1&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk9 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&maxBedrooms=4&minBedrooms=4&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolk10 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&minBedrooms=5&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
suffolkmostrecent = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&sortType=6&propertyTypes=&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords='
# Create a list of URLs
if False:
    urls = [ suffolk1, suffolk2, suffolk3, suffolk4, suffolk5, suffolk6, suffolk7, suffolk8, suffolk9, suffolk10]

    # Initialize an empty list to store the results
    results = []
    # Iterate over the URLs
    for url in urls:
        rm = RightmoveData(url)
        time.sleep(30)
        rm = rm.get_results
        results.append(rm)  # Append the RightmoveData instance itself

    # Concatenate the results into a single DataFrame
    all_results = pd.concat([result for result in results], ignore_index=True)

    # Save the concatenated results to a JSON file
    all_results.to_json('suffolk_all_data.json')

if False:
    rm = RightmoveData(suffolkmostrecent)
    rm.get_results.head(8).to_json('suffolkrecenthouselistings.json')

newhomesuffolk = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&sortType=6&propertyTypes=&includeSSTC=false&mustHave=newHome&dontShow=&furnishTypes=&keywords='
newhomesuffolk2 = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E61330&sortType=10&propertyTypes=&includeSSTC=false&mustHave=newHome&dontShow=&furnishTypes=&keywords='
if True:
    urls = [newhomesuffolk,newhomesuffolk2]

    # Initialize an empty list to store the results
    results = []
    # Iterate over the URLs
    for url in urls:
        rm = RightmoveData(url)
        time.sleep(30)
        rm = rm.get_results
        results.append(rm)  # Append the RightmoveData instance itself

    # Concatenate the results into a single DataFrame
    all_results = pd.concat([result for result in results], ignore_index=True)

    # Save the concatenated results to a JSON file
    all_results.to_json('suffolk_new_homes.json')
quit(1)

rm = RightmoveData(suffolkurl)
rm.get_results.to_json('suffolkdata.json')
quit(1)
soldprices = 'https://www.rightmove.co.uk/house-prices/woodbridge.html?country=england&searchLocation=Woodbridge&year=2'

# Create an instance of SoldProperties with the URL
sold_properties = SoldProperties(soldprices)

# Access the processed DataFrame
df = sold_properties.processed_data

# Display the DataFrame
df.to_json('sold_test.json')
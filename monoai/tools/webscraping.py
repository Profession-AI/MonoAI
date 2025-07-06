class WebScraping():

    """
    WebScraping is a tool that allows you to scrape the web.
    It uses the requests library to scrape the web.

    Examples
    --------

    ```python
    webscraping = WebScraping()
    result = webscraping.scrape("https://www.scrapethissite.com")
    print(result["html"]) # print the html of the page
    print(result["text"]) # print the content of the page merged into a single string
    ```
    """

    def __init__(self, engine:str = "requests", deep:bool = False):

        """
        Initialize the WebScraping tool.

        Parameters:
        ----------
        engine: str, optional
            The engine to use (requests, tavily, selenium. Default is requests)
        deep: bool, optional
            If using tavily, whether to use the advanced extraction mode (default is False)
        """

        if engine == "requests":
            self._engine = _RequestsScraper()
        elif engine == "tavily":
            self._engine = _TavilyScraper(deep=deep)
        elif engine == "selenium":
            self._engine = _SeleniumScraper()
        else:
            raise ValueError(f"Invalid engine: {engine} (must be 'requests', 'tavily', or 'selenium')")

    def scrape(self, url: str):

        """
        Scrape a webpage.

        Parameters:
        ----------
        url: str
            The url to scrape

        Returns:
        -------
        dict
            The response from the scraper.
            html: str
            The html of the page (not available if using tavily)
            text: str
            The content of the page merged into a single string
        """
        response, text_response = self._engine.scrape(url)
        return {"html": response, "text": text_response}
    

from bs4 import BeautifulSoup

class _BaseScraper():

    def __init__(self):
        pass

    def scrape(self, url: str):
        pass

    def _extract_text(self, content: str, remove_whitespace: bool = True) -> str:
        """
        Extract text content from HTML, optionally removing extra whitespace.
        
        Args:
            html_content (str): The HTML content to extract text from
            remove_whitespace (bool): Whether to remove extra whitespace and normalize spaces
            
        Returns:
            str: The extracted text content
        """
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        if remove_whitespace:
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text 


import requests

class _RequestsScraper(_BaseScraper):

    def __init__(self):
        super().__init__()

    def scrape(self, url: str):
        response = requests.get(url)
        return response.text, self._extract_text(response.text)
    

from tavily import TavilyClient
from monoai.keys.keys_manager import load_key

class _TavilyScraper(_BaseScraper):

    def __init__(self, deep:bool = False):
        super().__init__()
        load_key("tavily")
        self._client = TavilyClient()
        self._deep = deep

    def scrape(self, url: str):
        response = self._client.extract(url, extract_depth="advanced" if self._deep else "basic")
        response = response["results"][0]
        return None, response["raw_content"]

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

class _SeleniumScraper(_BaseScraper):
    """
    A scraper that uses Selenium to handle dynamic content and JavaScript.
    This is useful for websites that require JavaScript execution to load content.
    """
    
    def __init__(self, headless: bool = True, wait_time: int = 10):
        """
        Initialize the Selenium scraper.
        
        Args:
            headless (bool): Whether to run Chrome in headless mode (default: True)
            wait_time (int): Maximum time to wait for elements to load (default: 10 seconds)
        """
        super().__init__()
        self._wait_time = wait_time
        self._setup_driver(headless)
    
    def _setup_driver(self, headless: bool):
        """Set up the Chrome WebDriver with specified options."""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        self._driver = webdriver.Chrome(options=chrome_options)
        self._wait = WebDriverWait(self._driver, self._wait_time)
    
    def scrape(self, url: str):
        """
        Scrape a webpage using Selenium.
        
        Args:
            url (str): The URL to scrape
            
        Returns:
            tuple: (html_content, text_content)
        """
        try:
            self._driver.get(url)
            # Wait for the page to load
            time.sleep(2)  # Basic wait for initial page load
            
            # Get the page source after JavaScript execution
            html_content = self._driver.page_source
            text_content = self._extract_text(html_content)
            
            return html_content, text_content
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None, None
            
    def __del__(self):
        """Clean up the WebDriver when the object is destroyed."""
        if hasattr(self, '_driver'):
            self._driver.quit()

def scrape_web(url: str, engine: str = "requests", deep: bool = False):
    """Scrape a webpage using the specified engine.
    
    Args:
        url: The url to scrape
        engine: The engine to use (requests, tavily, selenium). Default is requests
        deep: If using tavily, whether to use the advanced extraction mode. Default is False
    
    Returns:
        A dictionary containing:
            html: The html of the page (not available if using tavily)
            text: The content of the page merged into a single string
            
    Raises:
        ValueError: If an invalid engine is specified
        
    Examples:
        Basic usage with requests engine:
        ```python
        result = scrape_web("https://www.scrapethissite.com")
        print(result["html"])  # print the html of the page
        print(result["text"])  # print the content of the page merged into a single string
        ```
        
        Using tavily with deep extraction:
        ```python
        result = scrape_web("https://example.com", engine="tavily", deep=True)
        print(result["text"])  # print the extracted content
        ```
    """
    
    if engine == "requests":
        scraper = _RequestsScraper()
    elif engine == "tavily":
        scraper = _TavilyScraper(deep=deep)
    elif engine == "selenium":
        scraper = _SeleniumScraper()
    else:
        raise ValueError(f"Invalid engine: {engine} (must be 'requests', 'tavily', or 'selenium')")
    
    response, text_response = scraper.scrape(url)
    return {"html": response, "text": text_response}


from bs4 import BeautifulSoup

class _BaseScraper():
    """Base class for web scrapers."""

    def __init__(self):
        """Initialize the base scraper."""
        pass

    def scrape(self, url: str):
        """Scrape a webpage.
        
        Args:
            url: The URL to scrape
            
        Returns:
            A tuple of (html_content, text_content)
        """
        pass

    def _extract_text(self, content: str, remove_whitespace: bool = True) -> str:
        """Extract text content from HTML, optionally removing extra whitespace.
        
        Args:
            content: The HTML content to extract text from
            remove_whitespace: Whether to remove extra whitespace and normalize spaces
            
        Returns:
            The extracted text content
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
    """Scraper that uses the requests library for basic web scraping."""

    def __init__(self):
        """Initialize the requests scraper."""
        super().__init__()

    def scrape(self, url: str):
        """Scrape a webpage using requests.
        
        Args:
            url: The URL to scrape
            
        Returns:
            A tuple of (html_content, text_content)
        """
        response = requests.get(url)
        return response.text, self._extract_text(response.text)
    

from tavily import TavilyClient
from monoai.keys.keys_manager import load_key

class _TavilyScraper(_BaseScraper):
    """Scraper that uses the Tavily API for advanced content extraction."""

    def __init__(self, deep: bool = False):
        """Initialize the Tavily scraper.
        
        Args:
            deep: Whether to use advanced extraction mode
        """
        super().__init__()
        load_key("tavily")
        self._client = TavilyClient()
        self._deep = deep

    def scrape(self, url: str):
        """Scrape a webpage using Tavily.
        
        Args:
            url: The URL to scrape
            
        Returns:
            A tuple of (None, extracted_content) since Tavily doesn't return raw HTML
        """
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
    """A scraper that uses Selenium to handle dynamic content and JavaScript.
    
    This is useful for websites that require JavaScript execution to load content.
    """
    
    def __init__(self, headless: bool = True, wait_time: int = 10):
        """Initialize the Selenium scraper.
        
        Args:
            headless: Whether to run Chrome in headless mode. Default is True
            wait_time: Maximum time to wait for elements to load in seconds. Default is 10
        """
        super().__init__()
        self._wait_time = wait_time
        self._setup_driver(headless)
    
    def _setup_driver(self, headless: bool):
        """Set up the Chrome WebDriver with specified options.
        
        Args:
            headless: Whether to run Chrome in headless mode
        """
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
        """Scrape a webpage using Selenium.
        
        Args:
            url: The URL to scrape
            
        Returns:
            A tuple of (html_content, text_content)
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

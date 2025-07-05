import os
import uuid
from typing import List, Dict, Tuple
from coicoi.tools.webscraping import WebScraping

class DocumentsBuilder:
    """
    A utility class for building document collections from various sources.
    
    This class provides methods to extract text content from files and web pages,
    split the content into manageable chunks with configurable size and overlap,
    and prepare the data for storage in vector databases.
    
    The DocumentsBuilder is designed to work seamlessly with the RAG system,
    producing output that can be directly used with vector database operations.
    
    Features:
    - File-based document extraction with UTF-8 encoding support
    - Web scraping with multiple engine options (requests, tavily, selenium)
    - Intelligent text chunking with word boundary preservation
    - Configurable chunk size and overlap parameters
    - Rich metadata generation for each document chunk
    - Unique ID generation for database storage
    
    Attributes:
        _chunk_size (int): Maximum size of each text chunk in characters
        _chunk_overlap (int): Number of characters to overlap between chunks
    
    Examples:
    --------
    Basic usage with file processing:
    
    ```python
    # Initialize with default chunk settings
    builder = DocumentsBuilder()
    
    # Process a text file
    documents, metadatas, ids = builder.from_file("document.txt")
    
    # Add to vector database
    vector_db.add(documents, metadatas, ids)
    ```
    
    Custom chunk settings:
    
    ```python
    # Initialize with custom chunk parameters
    builder = DocumentsBuilder(chunk_size=500, chunk_overlap=50)
    
    # Process multiple files
    for file_path in ["doc1.txt", "doc2.txt"]:
        documents, metadatas, ids = builder.from_file(file_path)
        vector_db.add(documents, metadatas, ids)
    ```
    
    Web scraping with different engines:
    
    ```python
    # Basic web scraping
    documents, metadatas, ids = builder.from_url("https://example.com")
    
    # Advanced scraping with Tavily
    documents, metadatas, ids = builder.from_url(
        "https://example.com",
        engine="tavily",
        deep=True
    )
    
    # JavaScript-heavy sites with Selenium
    documents, metadatas, ids = builder.from_url(
        "https://spa-example.com",
        engine="selenium"
    )
    ```
    
    Notes:
    ------
    - Chunk overlap helps maintain context between chunks
    - Word boundaries are preserved when possible to avoid cutting words
    - Empty chunks are automatically filtered out
    - All text is processed as UTF-8
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the DocumentsBuilder with chunking parameters.
        
        Parameters:
        -----------
        chunk_size : int, default=1000
            Maximum size of each text chunk in characters. Larger chunks
            preserve more context but may be less focused for retrieval.
            
        chunk_overlap : int, default=100
            Number of characters to overlap between consecutive chunks.
            Overlap helps maintain context and improves retrieval quality
            for queries that span chunk boundaries.
            
        Examples:
        ---------
        ```python
        # Default settings (good for most use cases)
        builder = DocumentsBuilder()
        
        # Small chunks for precise retrieval
        builder = DocumentsBuilder(chunk_size=500, chunk_overlap=50)
        
        # Large chunks for context preservation
        builder = DocumentsBuilder(chunk_size=2000, chunk_overlap=200)
        ```
        
        Notes:
        ------
        - chunk_overlap should typically be 10-20% of chunk_size
        - Very small chunks (< 200 chars) may lose context
        - Very large chunks (> 3000 chars) may be less focused for retrieval
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def from_file(self, file_path: str) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Read a file and split it into chunks with specified size and overlap.
        
        This method reads a text file from the filesystem, splits its content
        into chunks according to the configured parameters, and generates
        metadata and unique IDs for each chunk.
        
        Parameters:
        -----------
        file_path : str
            Path to the text file to read. The file must exist and be
            readable. UTF-8 encoding is assumed.
            
        Returns:
        --------
        Tuple[List[str], List[Dict], List[str]]
            A tuple containing:
            - List of document chunks (strings): The text content split into chunks
            - List of metadata dictionaries: Metadata for each chunk including
              file information and chunk details
            - List of unique IDs: UUID strings for each chunk
            
        Raises:
        -------
        FileNotFoundError
            If the specified file does not exist or is not accessible.
            
        UnicodeDecodeError
            If the file cannot be decoded as UTF-8.
            
        Examples:
        ---------
        ```python
        # Process a single file
        documents, metadatas, ids = builder.from_file("article.txt")
        
        # Access metadata information
        for i, metadata in enumerate(metadatas):
            print(f"Chunk {i+1}:")
            print(f"  File: {metadata['file_name']}")
            print(f"  Size: {metadata['chunk_size']} characters")
            print(f"  Position: {metadata['chunk_index'] + 1}/{metadata['total_chunks']}")
        ```
        
        Notes:
        ------
        - File is read entirely into memory before processing
        - Empty files will return empty lists
        - File path is stored in metadata for traceability
        - Chunk indexing starts at 0
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split text into chunks
        chunks = self._split_text(text)
        
        # Generate metadata and IDs for each chunk
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            
            # Create metadata
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk)
            }
            
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        return documents, metadatas, ids

    def from_url(self, url: str, engine: str = "requests", deep: bool = False) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Scrape content from a URL and split it into chunks with specified size and overlap.
        
        This method uses web scraping to extract text content from a webpage,
        then processes the content using the same chunking logic as file processing.
        Multiple scraping engines are supported for different types of websites.
        
        Parameters:
        -----------
        url : str
            The URL to scrape. Must be a valid HTTP/HTTPS URL.
            
        engine : str, default="requests"
            The web scraping engine to use:
            - "requests": Simple HTTP requests (fast, good for static content)
            - "tavily": Advanced web scraping with better content extraction
            - "selenium": Full browser automation (good for JavaScript-heavy sites)
            
        deep : bool, default=False
            If using the "tavily" engine, whether to use advanced extraction mode.
            Deep extraction provides better content quality but is slower.
            
        Returns:
        --------
        Tuple[List[str], List[Dict], List[str]]
            A tuple containing:
            - List of document chunks (strings): The scraped text split into chunks
            - List of metadata dictionaries: Metadata for each chunk including
              URL information and scraping details
            - List of unique IDs: UUID strings for each chunk
            
        Raises:
        -------
        ValueError
            If the scraping fails or no text content is extracted.
            
        Examples:
        ---------
        ```python
        # Basic web scraping
        documents, metadatas, ids = builder.from_url("https://example.com")
        
        # Advanced scraping with Tavily
        documents, metadatas, ids = builder.from_url(
            "https://blog.example.com",
            engine="tavily",
            deep=True
        )
        
        # JavaScript-heavy site with Selenium
        documents, metadatas, ids = builder.from_url(
            "https://spa.example.com",
            engine="selenium"
        )
        
        # Access scraping metadata
        for metadata in metadatas:
            print(f"Source: {metadata['url']}")
            print(f"Engine: {metadata['scraping_engine']}")
            print(f"Deep extraction: {metadata['deep_extraction']}")
        ```
        
        Notes:
        ------
        - Scraping may take time depending on the engine and website complexity
        - Some websites may block automated scraping
        - Selenium requires Chrome/Chromium to be installed
        - Tavily requires an API key to be configured
        """
        # Initialize WebScraping with specified engine
        scraper = WebScraping(engine=engine, deep=deep)
        
        # Scrape the URL
        result = scraper.scrape(url)
        
        if not result or not result.get("text"):
            raise ValueError(f"Failed to extract text content from URL: {url}")
        
        text = result["text"]
        
        # Split text into chunks
        chunks = self._split_text(text)
        
        # Generate metadata and IDs for each chunk
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            
            # Create metadata
            metadata = {
                'url': url,
                'source_type': 'web_page',
                'scraping_engine': engine,
                'deep_extraction': deep,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk)
            }
            
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        return documents, metadatas, ids
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with specified size and overlap.
        
        This private method implements the core text chunking algorithm.
        It attempts to break text at word boundaries to avoid cutting words
        in half, while respecting the configured chunk size and overlap parameters.
        
        Parameters:
        -----------
        text : str
            The text content to split into chunks.
            
        Returns:
        --------
        List[str]
            List of text chunks, each with maximum size equal to chunk_size.
            Empty chunks are automatically filtered out.
            
        Examples:
        ---------
        ```python
        # Internal usage (called by from_file and from_url)
        chunks = builder._split_text("This is a long text that needs to be split...")
        print(f"Created {len(chunks)} chunks")
        ```
        
        Notes:
        ------
        - Chunks are stripped of leading/trailing whitespace
        - Empty chunks are automatically filtered out
        - Word boundaries (spaces and newlines) are preferred break points
        - If no word boundary is found, text is broken at the exact size limit
        - The last chunk may be smaller than chunk_size
        - Overlap is applied between consecutive chunks
        """
        if len(text) <= self._chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position for current chunk
            end = start + self._chunk_size
            
            # If this is not the last chunk, try to break at a word boundary
            if end < len(text):
                # Look for the last space or newline before the end
                last_space = text.rfind(' ', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_space, last_newline)
                
                # If we found a good break point, use it
                if break_point > start:
                    end = break_point + 1
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            start = end - self._chunk_overlap
            if start >= len(text):
                break
        
        return chunks
        
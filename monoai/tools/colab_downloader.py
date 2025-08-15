import requests
import nbformat
from nbconvert import PythonExporter

def colab_downloader(colab_url):
    """
    Downloads a Jupyter notebook from Google Drive and returns its Python code as a string.

    Args:
        colab_url (str): The Google Drive URL of the notebook (should contain 'drive/').
    """
    
    BASE_URL = "https://docs.google.com/uc?export=download&id="
    try:
        file_id = colab_url.split("drive/")[1].split("?")[0]
    except Exception:
        return "Error: Invalid Google Drive URL."

    download_url = BASE_URL + file_id

    try:
        response = requests.get(download_url)
        response.raise_for_status()
        notebook_content = response.content
    except Exception as e:
        return f"Error downloading notebook: {e}"

    try:
        nb = nbformat.reads(notebook_content.decode('utf-8'), as_version=4)
        exporter = PythonExporter()
        source, _ = exporter.from_notebook_node(nb)
        return source
    except Exception as e:
        return f"Error converting notebook: {e}"




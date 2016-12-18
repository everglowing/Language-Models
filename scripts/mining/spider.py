import codecs
import requests
import re
from bs4 import BeautifulSoup

LANGUAGE = "te"
PAGE_LIMIT = 300
DEPTH = 10
FILE_NAME = "final_te.txt"

# Look up tables
languages = {
  "hi": ur"[^\u0900-\u097F]+",
  "ml": ur"[^\u0D00-\u0D7F\u002E]+",
  "ta": ur"[^\u0B80-\u0BFF\u002E]+",
  "kn": ur"[^\u0C80-\u0CFF\u002E]+",
  "te": ur"[^\u0C00-\u0C7F\u002E]+",
}
seed_urls = {
  "hi": "https://hi.wikipedia.org/wiki/%E0%A4%B8%E0%A5%8D%E0%A4%95%E0%A5%89%E0%A4%9F%E0%A5%8D%E0%A4%B2%E0%A5%88%E0%A4%A3%E0%A5%8D%E0%A4%A1",
  "ml": "https://ml.wikipedia.org/wiki/%E0%B4%97%E0%B5%8B%E0%B4%B5%E0%B4%AF%E0%B4%BF%E0%B4%B2%E0%B5%86_%E0%B4%AE%E0%B4%A4%E0%B4%A6%E0%B5%8D%E0%B4%B0%E0%B5%8B%E0%B4%B9%E0%B4%B5%E0%B4%BF%E0%B4%9A%E0%B4%BE%E0%B4%B0%E0%B4%A3%E0%B4%95%E0%B5%BE",
  "ta": "https://ta.wikipedia.org/wiki/%E0%AE%9A%E0%AE%AE%E0%AE%B0%E0%AF%8D%E0%AE%95%E0%AE%A8%E0%AF%8D%E0%AE%A4%E0%AF%81",
  "kn": "https://kn.wikipedia.org/wiki/%E0%B2%9C%E0%B3%86._%E0%B2%9C%E0%B2%AF%E0%B2%B2%E0%B2%B2%E0%B2%BF%E0%B2%A4%E0%B2%BE",
  "te": "https://te.wikipedia.org/wiki/%E0%B0%AA%E0%B1%8D%E0%B0%B2%E0%B0%BE%E0%B0%B8%E0%B1%80_%E0%B0%AF%E0%B1%81%E0%B0%A6%E0%B1%8D%E0%B0%A7%E0%B0%82",
}

# Used to remove image URLs
images = re.compile(r'\.(jpg|png|gif|bmp)$')

def fetch_webpage(url):
  """
  Download webpage using requests.
  Will return "" when invalid URL.
  """
  try:
    r = requests.get(url)
    r.raise_for_status()
  except Exception as err:
    print err
    return ""
  return r.text

def remove_peripherals(text, lang):
  """
  Used to remove all characters not in unicode block
  for that language.
  """
  return re.sub(languages[lang], " ", text)


class WikiWebpage(object):
  """
  Class to get specific data from Wikipedia article.
  Requires a Wikipedia URL for initialization.
  """
  def __init__(self, url, lang="en"):
    """
    Makes a BeautifulSoup object which has downloaded
    the webpage.
    """
    self.url = url
    self.lang = lang
    self.soup = BeautifulSoup(fetch_webpage(url), "lxml")

  def __str__(self):
    return "Wikipedia Webpage - " + self.url

  def get_child_urls(self):
    """
    Returns a list of all URLs of 'mw-redirect' class and
    on the same language domain.
    """
    # Not using regex as it is trivial here
    url_template = "/wiki/"
    child_urls = []
    for a in self.soup.findAll('a'):
      if a.get('href') and a.get('href').startswith(url_template) and \
         a.get('class', [None])[0] != 'mw-redirect':
        child_urls.append("https://" + self.lang + ".wikipedia.org" + a['href'])

    # filtering image files
    child_urls = filter(lambda x: images.search(x) is None, child_urls)
    return child_urls

  def get_data(self):
    """
    Gets a list of all the paragraph texts, stripped of HTML.
    """
    data = []
    for p in self.soup.find_all('p'):
      text = remove_peripherals(p.getText().strip(), self.lang)
      if text != "" and text.count("\n") == 0 and len(text.split()) > 10:
        data.append(text)
    return data


class Spider(object):
  """
  Class which makes objects of the WikiWebpage type and recursively
  iterates over the tree.
  """
  def __init__(self, seed_url, depth, lang, page_limit, filename="data.txt"):
    """
    seed_url: The initial url the crawler uses.
    depth: The maximum number of levels of the BFS.
    lang: Two alphabet lowercase language code.
    page_limit: Maximum number of URLs you wish to scan.
    filename: Output file name.
    """
    self.seed_url = seed_url
    self.depth = depth
    self.lang = lang
    self.urls = []
    self.data = ""
    self.page_limit = page_limit
    self.filename = filename

  def run(self, write_file=True):
    """
    Runs the crawler until depth is reached
    or page_limit is crossed.
    write_file: If True, file is written after each fetch.
    """
    self.data = ""
    current_urls = [self.seed_url]
    next_list = []
    # depth of BFS
    for i in range(0, self.depth):
      # Current depth's status
      print "Depth " + str(i) + " => " + str(len(current_urls)) + " URL(s)"

      for index, url in enumerate(current_urls):
        if url in self.urls:
          print "Repeat found!"
          continue
        # Build a WikiWebpage object which will do the scraping
        w = WikiWebpage(url, self.lang)
        # List of all URLs scanned
        self.urls.append(url)
        if (index+1) % 100 == 0:
          print "\t URLs scanned => " + str(index + 1)

        # Extract all the data
        for data in w.get_data():
          self.data += data + "\n"

        # Get all child urls
        child_urls = w.get_child_urls()
        # Filter out processed child URLs
        child_urls = filter(lambda x: x not in self.urls, child_urls)
        next_list.extend(child_urls)

        # Write file and clear data object
        if write_file:
          self.write_file()

        # Impose condition of total URLs
        if len(self.urls) >= self.page_limit:
          print "URL limit reached!"
          return

      # Remove duplicates URLs
      current_urls = list(set(next_list))

  def write_file(self):
    """
    Clear self.data and append the file specified
    in the Spider object.
    """
    with codecs.open(self.filename, "a", "utf8") as myfile:
      myfile.write(self.data)
    self.data = ""


def main():
  s = Spider(seed_url=seed_urls[LANGUAGE],
             depth=DEPTH,
             lang=LANGUAGE,
             page_limit=PAGE_LIMIT,
             filename=FILE_NAME)
  s.run(write_file=True)
  print s.data


if __name__ == "__main__":
  main()


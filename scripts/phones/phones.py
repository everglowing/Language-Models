import codecs
import requests
import re
from bs4 import BeautifulSoup

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

class PhoneWikiWebpage(object):
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

  def build_phones(self):
    """
    Returns a list of all URLs of 'mw-redirect' class and
    on the same language domain.
    """
    # Not using regex as it is trivial here
    url_template = "/wiki/"
    lists = {}
    counter = 0
    for a in self.soup.findAll('table'):
      if a.get('class', [None])[0] == 'wikitable' and self.lang in a.getText() and "IPA" in a.getText():
        for tr in a.findAll('tr'):
          cells = tr.findAll('td')
          if len(cells) < 2: continue
          for letter in cells[1].getText().split(','):
            if "[" in letter.strip():
              letter = letter.strip()[:-3]
            elif "-" in letter.strip():
              letter = letter.strip()[1:]
            else:
              letter = letter.strip()
            counter += 1
            lists[letter.strip()] = cells[0].getText().strip();
    return lists


def main():
  p = PhoneWikiWebpage("https://en.wikipedia.org/wiki/Help:IPA_for_Malayalam", "Malayalam")
  p = p.build_phones()
  p2 = PhoneWikiWebpage("https://en.wikipedia.org/wiki/Help:IPA_for_Tamil", "Tamil")
  p2 = p2.build_phones()
  tamil = ""
  for key,val in p2.iteritems():
    tamil += key + " " + val + "\n"
  malayalam = ""
  for key,val in p2.iteritems():
    malayalam += key + " " + val + "\n"
  with codecs.open('tamil_wiki.txt', 'w', 'utf-8') as f:
    f.write(tamil)
  with codecs.open('malayalam_wiki.txt', 'w', 'utf-8') as f:
    f.write(malayalam)

if __name__ == "__main__":
  main()
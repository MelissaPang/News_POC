import mediacloud.api
import datetime
import random
import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd

def create_mediacloud_client(api_key: str):
    """
    Establish connection via mediacloud api, get a key by signing up for an account
    :param api_key:
    :return mc:
    """
    mc = mediacloud.api.MediaCloud(my_key)
    return mc


def get_random_articles_within_timeframe(mc_client, start_date: datetime, end_date: datetime):
    """
    get random (topic ambivalent) english articles witin a given timeframe. Articles are in english and from top 5 US sources.
    Output is to write a csv with story_id, author, title, url, and text from story of given num_files
    :param mc_client: via create_mediacloud_client
    :param start_date: datetime format hardcode
    :param end_date:  datetime format hardcode
    :param num_files: must be integer
    :return: test dataset with given timeframe
    """
    query = "tags_id_media:58722749"
    #only look at US articles, no random selection - return number of articles in a day
    #select only specific medias 
    date_range = mc_client.dates_as_query_clause(start_date, end_date)
    stories = mc_client.storyList(query, date_range, rows=10000)

    fieldnames = ['stories_id', 'publish_date', 'title', 'url', 'language', 'ap_syndicated', 'themes', 'media_id',
                  'media_name', 'media_url', 'text']
    with open('sample_articles.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for s in stories:
            url = s['url']
            head = requests.head(url)
            head = str(head)
            if head == "<Response [400]":
                pass
            else:
                res = requests.get(url)
                html_page = res.content
                soup = BeautifulSoup(html_page, 'html.parser')
                # getting all the paragraphs
                output = ''
                for para in soup.find_all("p", text=True):
                    p = para.get_text()
                    output += '{} '.format(p)
                s['text'] = output
                writer.writerow(s)



def get_text_from_url(df):
    # Nice to have if there's time
    # select articles from x media sources
    # check for dead links?
    # read in CSV

    for s in df['stories_id']:
        url = df[s]['url']
        res = requests.get(url)
        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        text = soup.find_all(text=True)
        output = ''
        blacklist = [
            '[document]',
            'noscript',
            'header',
            'html',
            'meta',
            'head',
            'input',
            'script',
            # there may be more elements you don't want, such as "style", etc.
        ]
        for t in text:
            if t.parent.name not in blacklist:
                output += '{} '.format(t)
                s['new_text'] = output


#key from free public account created with personal email
my_key = '2b9ac182814bebf7481aa953e91f55a97f87df0e3ce9a5fa05159333a4f55eb4'

# date parameters: 2021/3/22 - 2021/3/29 chosen as the time frame wherein the Ever Green got stuck/unstuck in that canal
start_date = datetime.date(2021, 3, 23)
end_date = datetime.date(2021, 3, 29)

mc = create_mediacloud_client(my_key)
get_random_articles_within_timeframe(mc, start_date, end_date)

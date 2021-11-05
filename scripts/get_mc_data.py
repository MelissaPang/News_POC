import os, mediacloud.api
from dotenv import load_dotenv
import datetime
import mediacloud.tags
import csv


load_dotenv()  # load config from .env file
mc = mediacloud.api.MediaCloud(os.getenv('MC_API_KEY'))
mediacloud.__version__

kat_query = '"hurricane katrina"'
mc.storyCount(kat_query,'publish_day:[2005-01-01T00:00:00Z TO 2006-01-01T00:00:00Z]')
start_date = datetime.date(2004,1,1)
end_date = datetime.date(2006,1,1)
date_range = mc.dates_as_query_clause(start_date, end_date)
results = mc.storyCount(kat_query, date_range, split=True, split_period="month")
print(results)

def all_matching_stories(mc_client, q, fq):
    """
    Return all the stories matching a query within Media Cloud. Page through the results automatically.
    :param mc_client: a `mediacloud.api.MediaCloud` object instantiated with your API key already
    :param q: your boolean query
    :param fq: your date range query
    :return: a list of media cloud story items
    """
    last_id = 0
    more_stories = True
    stories = []
    while more_stories:
        page = mc_client.storyList(q, fq, last_processed_stories_id=last_id, rows=500, sort='processed_stories_id')
        print("  got one page with {} stories".format(len(page)))
        if len(page) == 0:
            more_stories = False
        else:
            stories += page
            last_id = page[-1]['processed_stories_id']
    return stories

katrina_query = mc.dates_as_query_clause(datetime.date(2004,1,1), datetime.date(2006,1,1))
all_stories = all_matching_stories(mc, kat_query, katrina_query)
len(all_stories)

for s in all_stories:
    # see the "language" notebook for more details on themes
    theme_tag_names = ','.join([t['tag'] for t in s['story_tags'] if t['tag_sets_id'] == mediacloud.tags.TAG_SET_NYT_THEMES])
    s['themes'] = theme_tag_names
# now write the CSV

fieldnames = ['stories_id', 'publish_date', 'title', 'url', 'language', 'ap_syndicated', 'themes', 'media_id', 'media_name', 'media_url']
with open('story-list.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for s in all_stories:
        writer.writerow(s)
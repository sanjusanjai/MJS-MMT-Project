from youtubesearchpython import VideosSearch
from pprint import pprint


videosSearch = VideosSearch('Hey, Soul Sister by train', limit = 1)

pprint(videosSearch.result())
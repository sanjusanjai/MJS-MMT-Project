import csv
from bs4 import BeautifulSoup
import requests
years=list(range(2006, 2022))
# Make a request to the webpage
for year in years:
    url=f"https://www.billboard.com/charts/year-end/"+str(year)+"/hot-100-songs/"
    
    print(url)
    response = requests.get(url)
    # print(response.text)

    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract data from the HTML structure
    results = soup.find_all('div', class_='o-chart-results-list-row-container')

    # print(results[0].prettify())

    # title=results[0].find('h3', id='title-of-a-story').text.strip()
    # # artist=results[0].find('span', class_="c-label a-font-primary-s").text.strip()
    # # code to extract class="c-label a-font-primary-s" from the HTML which is inside the span tag
    # artist_and_rank=results[0].find_all('span', class_='c-label')
    # rank=artist_and_rank[0].text.strip()
    # artist=artist_and_rank[1].text.strip()



    # # print title
    # print("Title:", title)
    # # print artist
    # print("Artist:", artist)
    # # print rank
    # print("Rank:", rank)
    with open(f'billboard_{year}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["rank","title", "artist"])

    for result in results:
        try:
            title = result.find('h3', id='title-of-a-story').text.strip()
        except AttributeError as e:
            title = None
        try:
            artist_and_rank=result.find_all('span', class_='c-label')
            rank=artist_and_rank[0].text.strip()
            artist=artist_and_rank[1].text.strip()
        except AttributeError as e:
            artist_and_rank = None
            artist=None
            rank=None
        # artist_and_rank=soup.find_all('span', class_='c-label')
        # rank=artist_and_rank[0].text.strip()
        # artist=artist_and_rank[1].text.strip()
        # print("Title:", title)
        # print("Artist:", artist)
        # print("Rank:", rank)
        # write all thes data to a csv file
        with open(f'billboard_{year}.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([rank,title, artist])
import csv
import requests
from bs4 import BeautifulSoup

# Make a request to the webpage
# Replace with the URL of the webpage containing the table
url = 'https://myswar.co/popular_song/year/2020/1'
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find all elements with class 'song_detail_display_table'
tables = soup.find_all('table', class_='song_detail_display_table')

# Extract data from each table and store in a list
data = []
for table in tables:
    print(table)
    # Extract song title
    song_title = table.find(
        'a', class_='songs_like_this2').text.strip()  # works
    print(song_title)

    # Extract the album name
    album_name = table.find('span', class_='attribute_value').a.text.strip()
    print(album_name)

    # Extract the singer name
    singer_name = table.find_all('span', class_='attribute_value')[
        1].a.text.strip()
    print(singer_name)

    # Extract the music director name̥
    music_director_name = table.find_all('span', class_='attribute_value')[
        2].a.text.strip()
    print(music_director_name)

    # Extract the lyrics writer name
    lyrics_writer_name = table.find_all('span', class_='attribute_value')[
        3].a.text.strip()
    print(lyrics_writer_name)

    # Extract the genre name
    try:
        genre_name = table.find_all('span', class_='attribute_value')[
            4].a.text.strip()
        print(genre_name)
    except:
        genre_name = None
        print(genre_name)

    # Extract the link to the YouTube video
    youtube_link = table.find(
        'a', href=True, target='_blank', rel='nofollow').get('href')
    print(youtube_link)
    # break

    # Extract the link to the Apple Music song
    try:
        apple_music_link = table.find(
            'a', href=True, title='Apple Music', target='_blank', rel='nofollow').get('href')
        print(apple_music_link)
    except:
        apple_music_link = None
        print(apple_music_link)
    # apple_music_link = table.find('a', href=True, title='Apple Music', target='_blank', rel='nofollow').get('href')

    # Extract the link to the iTunes song
    try:
        itunes_link = table.find(
            'a', href=True, title='iTunes', target='_blank', rel='nofollow').get('href')
        print(itunes_link)
    except:
        itunes_link = None
        print(itunes_link)
    # itunes_link = table.find('a', href=True, title='iTunes', target='_blank', rel='nofollow').get('href')

    # Append extracted data to the list
    data.append([song_title, album_name, singer_name, music_director_name,
                lyrics_writer_name, genre_name, youtube_link, apple_music_link, itunes_link])

# Write the extracted data to a CSV file
filename = 'song_details_v2.csv'  # Specify the filename for the CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Song Title', 'Album Name', 'Singer Name', 'Music Director Name',
                     'Lyrics Writer Name', 'Genre Name', 'YouTube Link', 'Apple Music Link', 'iTunes Link'])
    writer.writerows(data)

print(f'Success! Extracted data has been written to {filename}.')

import csv
import requests
from bs4 import BeautifulSoup

# Specify the years and page numbers to scrape
years = [2020, 2021]  # Add more years if needed
years = list(range(2000, 2022))
page_numbers = list(range(1,20))  # Add more page numbers if needed

for year in years:
	for page_number in page_numbers:
		# Make a request to the webpage
		url = f'https://myswar.co/popular_song/year/{year}/{page_number}'
		response = requests.get(url)
		#check if the page exists
		if response.status_code == 404:
			break

		# Parse the HTML content
		soup = BeautifulSoup(response.content, 'html.parser')

		# Find all elements with class 'song_detail_display_table'
		tables = soup.find_all('table', class_='song_detail_display_table')

		# Extract data from each table
		data = []
		
		# add header row
		header_row = ['song_title', 'album_name', 'singer_name', 'music_director_name', 'lyrics_writer_name', 'genre_name', 'youtube_link']
		data.append(header_row)


		for table in tables:
			# Extract song title
			try:
				song_title = table.find(
					'a', class_='songs_like_this2').text.strip()  # works
			except:
				song_title = None
			# song_title = table.find(
			# 	'a', class_='songs_like_this2').text.strip()  # works
			# print(song_title)

			# Extract the album name
			try:
				album_name=table.find('span', class_='attribute_value').a.text.strip()
			except:
				album_name=None
			# album_name = table.find('span', class_='attribute_value').a.text.strip()
			# print(album_name)

			# Extract the singer name
			try:
				singer_name=table.find_all('span', class_='attribute_value')[
				1].a.text.strip()
			except:
				singer_name=None
			# singer_name = table.find_all('span', class_='attribute_value')[
			# 	1].a.text.strip()
			# print(singer_name)

			# Extract the music director name̥
			try:
				music_director_name = table.find_all('span', class_='attribute_value')[
					2].a.text.strip()
				# print(music_director_name)
			except:
				music_director_name = None
				# print(music_director_name)
			# music_director_name = table.find_all('span', class_='attribute_value')[
			# 	2].a.text.strip()
			# print(music_director_name)

			# Extract the lyrics writer name
			try:
				lyrics_writer_name = table.find_all('span', class_='attribute_value')[
					3].a.text.strip()
				# print(lyrics_writer_name)
			except:
				lyrics_writer_name = None
				# print(lyrics_writer_name)
			# lyrics_writer_name = table.find_all('span', class_='attribute_value')[
				# 3].a.text.strip()
			# print(lyrics_writer_name)

			# Extract the genre name
			try:
				genre_name = table.find_all('span', class_='attribute_value')[
					4].a.text.strip()
				# print(genre_name)
			except:
				genre_name = None
				# print(genre_name)

			# Extract the link to the YouTube video
			try:
				youtube_link = table.find(
					'a', href=True, target='_blank', rel='nofollow').get('href')
				# print(youtube_link)
			except:
				youtube_link = None
				# print(youtube_link)
			# youtube_link = table.find(
			# 	'a', href=True, target='_blank', rel='nofollow').get('href')
			# print(youtube_link)
			# break

			# Extract the link to the Apple Music song
			try:
				apple_music_link = table.find(
					'a', href=True, title='Apple Music', target='_blank', rel='nofollow').get('href')
				# print(apple_music_link)
			except:
				apple_music_link = None
				# print(apple_music_link)
			# apple_music_link = table.find('a', href=True, title='Apple Music', target='_blank', rel='nofollow').get('href')

			# Extract the link to the iTunes song
			try:
				itunes_link = table.find(
					'a', href=True, title='iTunes', target='_blank', rel='nofollow').get('href')
				# print(itunes_link)
			except:
				itunes_link = None
				# print(itunes_link)
			# itunes_link = table.find('a', href=True, title='iTunes', target='_blank', rel='nofollow').get('href')

			# Append extracted data to the list
			data.append([song_title, album_name, singer_name, music_director_name,
						lyrics_writer_name, genre_name, youtube_link])
			#convert all None of data to empty string
			# print(data)
			data = [[x if x is not None else '' for x in row] for row in data]
			# break
	  # Write the extracted data to a CSV file
		filename = f'song_details_{year}_page{page_number}.csv'
		with open(filename, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerows(data)

		print(
			f'Success! Extracted data for {year}, Page {page_number} has been written to {filename}.')

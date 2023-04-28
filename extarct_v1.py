import csv
import requests
from bs4 import BeautifulSoup

# Make a request to the webpage
url = 'https://myswar.co/popular_song/year/2020/1'  # Replace with the URL of the webpage containing the table
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find all elements with class 'song_detail_display_table'
tables = soup.find_all('table', class_='song_detail_display_table')

# Extract data from each table
data = []
for table in tables:
    row_data = []
    # Extract data from each cell in the table
    for row in table.find_all('tr'):
        cell_data = []
        for cell in row.find_all(['td', 'th']):
            cell_data.append(cell.text.strip())
        row_data.append(cell_data)
    data.append(row_data)

# Write the extracted data to a CSV file
filename = 'song_details.csv'  # Specify the filename for the CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for table_data in data:
        writer.writerows(table_data)

print(f'Success! Extracted data has been written to {filename}.')

import requests
import csv
from bs4 import BeautifulSoup

# URL of the webpage you want to scrape
years = list(range(2000, 2022))
# years=[2010]
for year in years:
    url = f"https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}"
    # Send an HTTP GET request to the URL and fetch the HTML content
    response = requests.get(url)
    html = response.content
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    

    # Find all <tr> elements within the <table> element
    rows = soup.find_all("tr")

    # Extract the data from each row
    with open(f"wki_{year}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["rank", "title", "artist"])

    count = 0

    for row in rows:
        # Skip the first row
        if count == 0:
            count += 1
            continue
        # Extract the text from the <td> elements within the row
        data = [
            td.get_text(strip=True).replace('''"''', "") for td in row.find_all("td")
        ]
        # Print the extracted data
        print(data)
        # Write the extracted data to a CSV file

        # write until count=100
        if count == 100:
            break
        with open(f"wiki_{year}.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(data)

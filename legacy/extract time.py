from bs4 import BeautifulSoup

file_path = "News | Yle Uutiset.html"
try:
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 2. Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # 3. Find all <time> tags as per the file structure
    time_tags = soup.find_all('time')

    print(f"--- Found {len(time_tags)} Article Times ---")
    
    # 4. Loop through and print each time
    for tag in time_tags:
        # get_text() extracts the time (e.g., "10:31")
        print(f"Published at: {tag.get_text().strip()}")

except FileNotFoundError:
    print(f"Error: Could not find '{file_path}'. Make sure the name matches exactly!")
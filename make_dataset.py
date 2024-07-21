import requests
from bs4 import BeautifulSoup
import openai
from tenacity import retry, stop_after_attempt, retry_if_exception_type, RetryError
from datetime import datetime
import pytz
from icml_finder.data import Session
import json
import jsonlines
from tqdm.auto import tqdm
from huggingface_hub import HfApi

api = HfApi()
client = openai.OpenAI()


# Function to parse time using OpenAI's GPT-3.5-turbo with retries and convert to Vienna time zone
@retry(stop=stop_after_attempt(1), retry=retry_if_exception_type(ValueError))
def parse_time_with_gpt(time_string):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant responding with JSON",
            },
            {
                "role": "user",
                "content": f"Extract the time and date from this string: {time_string} the format should be a json dict with {{'datetime': Year-Month-Date Hour:Minute Timezone}} Assume it is July 2024. Dont add any  ```json ```",
            },
        ],
    )
    try:
        time_json = json.loads(response.choices[0].message.content)
        date_string = time_json["datetime"]

        # Removing the 'PDT' from the string for parsing
        date_string_parsed = date_string[:-4].strip()

        # Parsing the datetime string to a datetime object
        dt_naive = datetime.strptime(date_string_parsed, "%Y-%m-%d %H:%M")

        # Define the Pacific time zone with daylight saving time
        pacific_tz = pytz.timezone("America/Los_Angeles")

        # Localize the naive datetime
        dt_pacific = pacific_tz.localize(dt_naive)

        # Define the Vienna time zone
        vienna_tz = pytz.timezone("Europe/Vienna")

        # Convert to Vienna time zone
        dt_vienna = dt_pacific.astimezone(vienna_tz)

        return dt_vienna.isoformat()  # Convert to ISO format for JSON serialization

    except Exception as e:
        raise ValueError(f"Error parsing and converting time with GPT: {e}")


def make_embedding(content: str):
    return (
        client.embeddings.create(
            input=[content.strip()], model="text-embedding-3-large"
        )
        .data[0]
        .embedding
    )


# Function to scrape location and time from the HTML content
def scrape_location_time(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    location = soup.find("h5", class_="text-center text-muted")
    location = location.get_text(strip=True) if location else "Unknown Location"

    time_info = soup.find("div", class_="text-center p-4").text

    try:
        vienna_time = parse_time_with_gpt(time_info)
    except RetryError:
        utc = pytz.timezone("UTC")
        vienna_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=utc)

    return location, vienna_time


if __name__ == "__main__":
    with open("data/ICML 2024 Events.json", "r") as f:
        json_data = json.load(f)

    for item in json_data:
        if "speakers/authors" in item:
            item["speakers_authors"] = item.pop("speakers/authors")

    updated_sessions = []

    for session_data in tqdm(json_data, desc="Processing Sessions"):
        location, vienna_time = scrape_location_time(session_data["virtualsite_url"])
        embedding_content = (
            f'Title: {session_data["name"]}; Abstract: {session_data["abstract"]}'
        )
        embedding = make_embedding(embedding_content)
        data = {
            "type": session_data["type"],
            "name": session_data["name"],
            "virtualsite_url": session_data["virtualsite_url"],
            "speakers_authors": session_data["speakers_authors"],
            "abstract": session_data["abstract"],
            "location": location,
            "time_vienna": vienna_time,
            "embedding": list(embedding),
        }
        updated_sessions.append(Session(**data))

    # Writing to a JSON Lines file
    with jsonlines.open("data/icml_sessions.jsonl", mode="w") as writer:
        for session in updated_sessions:
            writer.write(session.model_dump_json())

    api.upload_file(
        path_or_fileobj="data/icml_sessions.jsonl",
        path_in_repo="icml_sessions.jsonl",
        repo_id="porestar/icml2024_embeddings",
        repo_type="dataset",
    )

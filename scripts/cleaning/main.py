import re
import pandas as pd
import html

INPUT_PATH = "../../data/raw_Twitter/"
OUTPUT_PATH = "../../data/clean_Twitter/"

show_debug = True
save_to_file = True

URL_RE = re.compile(r"https?://\S+")
PUNCT_RE = re.compile(r"[^\w\s\$\%]")
MENTION_RE  = re.compile(r"(?<!\$)@\w+")
HASHTAG_RE  = re.compile(r"#(\w+)")
EMOJI_RE    = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

def clean_Tweet(tweet: str):
    if not isinstance(tweet, str):
        return ""

    #output = html.unescape(tweet) 
    output = URL_RE.sub(" ", tweet)
    output = MENTION_RE.sub(" ", output)
    output = HASHTAG_RE.sub(r"\1", output)
    output = PUNCT_RE.sub(" ", output)
    output = EMOJI_RE.sub(" ", output)
    output = output.lower()
    output = WHITESPACE_RE.sub(" ", output)
    output = output.strip()
    return output

def main():
    jan_data = pd.read_csv(f"{INPUT_PATH}FT_Jan_2020.csv")
    feb_data = pd.read_csv(f"{INPUT_PATH}FT_Feb_2020.csv")
    mar_data = pd.read_csv(f"{INPUT_PATH}FT_Mar_2020.csv")
    apr_data = pd.read_csv(f"{INPUT_PATH}FT_Apr_2020.csv")
    may_data = pd.read_csv(f"{INPUT_PATH}FT_May_2020.csv")
    jun_data = pd.read_csv(f"{INPUT_PATH}FT_Jun_2020.csv")
    jul_data = pd.read_csv(f"{INPUT_PATH}FT_Jul_2020.csv")
    aug_data = pd.read_csv(f"{INPUT_PATH}FT_Aug_2020.csv")
    sep_data = pd.read_csv(f"{INPUT_PATH}FT_Sep_2020.csv")
    oct_data = pd.read_csv(f"{INPUT_PATH}FT_Oct_2020.csv")
    nov_data = pd.read_csv(f"{INPUT_PATH}FT_Nov_2020.csv")
    dec_data = pd.read_csv(f"{INPUT_PATH}FT_Dec_2020.csv")

    year_data = [jan_data, feb_data, mar_data, apr_data, may_data, jun_data, jul_data, aug_data, sep_data, oct_data, nov_data, dec_data]

    clean_year_data = []

    for month in year_data:
        month["clean_text"] = month["tweet"].apply(clean_Tweet)
        month = month[["created_at", "clean_text"]].copy()
        month = month.rename(columns={"created_at": "timestamp", "clean_text": "text"})
        month = month.drop_duplicates(subset=["text"], keep="first")
        clean_year_data.append(month)

    all_tweets = pd.concat(clean_year_data, ignore_index=True)
    
    if show_debug:
        print(all_tweets)
    
    if save_to_file:
        all_tweets.to_csv(f"{OUTPUT_PATH}clean_tweets_2020.csv", index=False)

if __name__ == "__main__":
    main()
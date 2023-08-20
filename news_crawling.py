import os
import sys
import urllib.request
import urllib.parse
import pandas as pd
import json

client_id = "1YEvn1FkqNyqZWPvPqIV"
client_secret = "zIkv32MySS"
encText = urllib.parse.quote("애플 주가")
display = 100
total_result = []

startDate = "2021-01-01"
endDate = "2023-01-01"

for start in range(1, 1001, 100):
    url = f"https://openapi.naver.com/v1/search/news.json?query={encText}&display={display}&start={start}&sort=sim&startDate={startDate}&endDate={endDate}"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if rescode == 200:
        response_body = response.read()
        total_result.append(response_body.decode('utf-8'))
    else:
        print("Error Code:", rescode)
    result = response_body.decode('utf-8')
    print(type(result))
    news_result = json.loads(result)
    print(news_result)
    df = pd.DataFrame(news_result['items'])
    df[['title']].to_csv("news.csv", mode='a', header=False)
print("".join(total_result))
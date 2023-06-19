import streamlit as st
import requests as rq
import datetime
import urllib.parse
import pytz
import base64

CLOUDAPI_BASE = "https://rest.litemoment.com"
THREE_HOURS_IN_SECONDS = 3 * 60 * 60
youtube_timeline = ""
MAX_NUMBER_RESULTS = 100
def generate_timeline(username, password, date, hour, minutes, seconds):
    youtube_timeline = ""
    timezone = pytz.timezone("America/Los_Angeles")
    absolute_time = datetime.datetime(date.year, date.month, date.day, hour, minutes, seconds)
    absolute_time_tz = timezone.localize(absolute_time)
    timestamp = int(absolute_time_tz.timestamp())
    condition = "/events?sort=-eventTS&max_results=" + str(MAX_NUMBER_RESULTS)
    b = (base64.b64encode(bytes(username + ":" + password,"utf-8"))).decode("utf-8")
    header = {"Authorization" : "Basic " + b}
    where = '{"$and":[{"eventTS":{"$gte":'+ str(timestamp) +'}},{"eventTS":{"$lte":' + str(timestamp + THREE_HOURS_IN_SECONDS) + "}}]}"
    print(where)
    where = urllib.parse.quote(where)
    print(where)
    url = CLOUDAPI_BASE + condition + "&where=" + where
    print(url)
    #get response
    response = rq.get(url, headers=header)
    events = response.json()["_items"]
    litesArray = []
    for lite in events:
        eventType = lite["eventType"]
        eventTS = lite["eventTS"]
        diff = eventTS - timestamp
        if diff < 0:
            continue
        litesArray.append((str(datetime.timedelta(seconds = diff)), eventType))
        litesArray = list(set(i for i in litesArray))
        litesArray.sort(key = lambda x : x[0])
    for t, a in litesArray:
        youtube_timeline += t + " " + a + "\n"
    st.code(youtube_timeline, language="python")
    return youtube_timeline
st.set_page_config(page_title = "Lite_timestamp", layout = "centered")
st.subheader("Litemoment Youtube timestamp generator")
st.write("---")
user_in, pass_in = st.columns(2)
with user_in:
    username = st.text_input("Enter username")
with pass_in:
    password = st.text_input("Enter password", type="password")
st.write("---")
st.write("Please enter the start time of the video:")
date_in, hour_in, minutes_in, seconds_in = st.columns(4)
with date_in:
    date = st.date_input("Date:")
with hour_in:
    hour = st.number_input("Hour:", step = 1, min_value = 0, max_value = 23, on_change = None)
with minutes_in:
    minutes = st.number_input("Minutes:", step = 1, min_value = 0, max_value = 59, on_change = None)
with seconds_in:
    seconds = st.number_input("Seconds:", step = 1, min_value = 0, max_value = 59, on_change = None)
if st.button("Generate timestamps", key="generate_timestamp"):
    youtube_timeline = generate_timeline(username, password, date, hour, minutes, seconds)



#form the url

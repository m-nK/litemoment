import streamlit as st
import requests as rq
import datetime
import urllib.parse
import pytz
import base64

CLOUDAPI_BASE = "https://rest.litemoment.com"
THREE_HOURS_IN_SECONDS = 3 * 60 * 60
MAX_NUMBER_RESULTS = 100
HINDSIGHT_IN_SECONDS = 15
#________________________________________functions______________________________________________________
def generate_timeline(username, password, date, hour, minutes, seconds, with_event_type):
    # youtube_timeline = ""
    timezone = pytz.timezone("America/Los_Angeles")
    try:
        absolute_time = datetime.datetime(date.year, date.month, date.day, hour, minutes, seconds)
    except:
        st.error("invalid time input")
        return
    absolute_time_tz = timezone.localize(absolute_time)
    timestamp = int(absolute_time_tz.timestamp())
    condition = "/events?sort=-eventTS&max_results=" + str(MAX_NUMBER_RESULTS)
    b = (base64.b64encode(bytes(username + ":" + password,"utf-8"))).decode("utf-8")
    header = {"Authorization" : "Basic " + b}
    where = '{"$and":[{"eventTS":{"$gte":'+ str(timestamp) +'}},{"eventTS":{"$lte":' + str(timestamp + THREE_HOURS_IN_SECONDS) + "}}]}"
    where = urllib.parse.quote(where)
    url = CLOUDAPI_BASE + condition + "&where=" + where
    #get response
    response = rq.get(url, headers=header)
    if response.status_code != 200:
        st.error("Wrong Username or Password")
        return
    try: 
        events = response.json()["_items"]
    except:
        st.error("invalid time input")
        return
    litesArray = []
    for lite in events:
        eventType = lite["eventType"]
        eventTS = lite["eventTS"]
        #apply hindsight offset
        diff = eventTS - timestamp - HINDSIGHT_IN_SECONDS
        if diff < 0:
            diff = 0
        litesArray.append((str(datetime.timedelta(seconds = diff)), eventType))
    litesArray = list(set(i for i in litesArray))
    litesArray.sort(key = lambda x : x[0])
    for t, a in litesArray:
        st.session_state.timeline += t + " " + a + "\n" if with_event_type else t + "\n"
#________________________________________start_of_homepage______________________________________________________
st.set_page_config(page_title = "Hindsight Seconds", layout = "wide")
st.title("Hindsight Seconds")
# st.write("---")
st.markdown("""
    1. Enter Litemoment credentials.
    2. Enter the start time of the recorded video.
    3. Click generate!
    """)
if "show_timeline" not in st.session_state:
    st.session_state.show_timeline = False
if "timeline" not in st.session_state:
    st.session_state.timeline = ""
with st.sidebar:
    username = st.text_input("Enter username")
    password = st.text_input("Enter password", type="password")
    st.write("---")
    st.write("Please enter the start time of the video:")
    date = st.date_input("Date:")
    hour = st.number_input("Hour:", step = 1, min_value = 0, max_value = 23, on_change = None)
    minutes = st.number_input("Minutes:", step = 1, min_value = 0, max_value = 59, on_change = None)
    seconds = st.number_input("Seconds:", step = 1, min_value = 0, max_value = 59, on_change = None)
    with_event_type = st.checkbox("Include Event Type", key="include_eventtype", on_change = None)
    st.write("##")
    generate = st.button("Generate timestamps", key="generate_timestamp")
if generate:
    st.session_state.timeline = ""
    generate_timeline(username, password, date, hour, minutes, seconds, with_event_type)
    st.session_state.show_timeline = True
if st.session_state.show_timeline and st.session_state.timeline:
    st.code(st.session_state.timeline, language="python")




#form the url

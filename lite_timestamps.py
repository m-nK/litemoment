import streamlit as st
import requests as rq
import datetime
import urllib.parse
import pytz
import base64

CLOUDAPI_BASE = "https://rest.litemoment.com"
THREE_HOURS_IN_SECONDS = 3 * 60 * 60
MAX_NUMBER_RESULTS = 100
DEFAULT_MAP = {"LITE":"LITE", "VAR":"VAR", "HADD":"HADD"}
#________________________________________functions______________________________________________________
def generate_timeline(username, password, date, hour, minutes, seconds, with_event_type, hindsight):
    timezone = pytz.timezone("America/Los_Angeles")
    try:
        absolute_time = datetime.datetime(date.year, date.month, date.day, hour, minutes, seconds)
    except:
        st.error("invalid time input")
        return
    if hindsight < 0 or hindsight > 60:
        st.error("Undefined Value")
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
        diff = eventTS - timestamp - hindsight
        if diff < 0:
            diff = 0
        litesArray.append((str(datetime.timedelta(seconds = diff)), eventType))
    litesArray = list(set(i for i in litesArray))
    litesArray.sort(key = lambda x : x[0])
    for t, a in litesArray:
        mapped = st.session_state.event_type_map[a] if a in st.session_state.event_type_map else ""
        st.session_state.timeline += t + " " + mapped + "\n" if with_event_type else t + "\n"
#________________________________________start_of_homepage______________________________________________________
st.set_page_config(page_title = "Hindsight Seconds", layout = "wide")
st.title("Litemoment Youtube Timestamp Generator")
st.markdown("""
    1. Enter Litemoment credentials.
    2. Enter the start time of the recorded video.
    3. Click generate!
    """)
#initialize parameters
if "show_timeline" not in st.session_state:
    st.session_state.show_timeline = False
if "timeline" not in st.session_state:
    st.session_state.timeline = ""
if "event_type_map" not in st.session_state:
    st.session_state.event_type_map = DEFAULT_MAP
#sidebar
with st.sidebar:
    username = st.text_input("Enter username")
    password = st.text_input("Enter password", type="password")
    st.write("---")
    st.write("Please enter the start time of the video:")
    date = st.date_input("Date:")
    hour = st.number_input("Hour:", step = 1, min_value = 0, max_value = 23)
    minutes = st.number_input("Minutes:", step = 1, min_value = 0, max_value = 59)
    seconds = st.number_input("Seconds:", step = 1, min_value = 0, max_value = 59)
    hindsight = st.number_input("Hindsight Seconds:", min_value = 5, max_value = 60, value = 15, step = 5)
    with_event_type = st.checkbox("Include Event Type", key="include_eventtype", value = True)
    if st.session_state.include_eventtype:
        "Translate Event Type"
        col1, col2, col3 = st.columns(3)
        with col1:
            new_lite = st.text_input("LITE:", value = st.session_state.event_type_map["LITE"])
        with col2:
            new_var = st.text_input("VAR:", value = st.session_state.event_type_map["VAR"])
        with col3:
            new_hadd = st.text_input("HADD:", value = st.session_state.event_type_map["HADD"])
        change_col, _, reset_col = st.columns(3)
        with change_col:
            change_button = st.button("Change", key="change_button", use_container_width = True)
        with reset_col:
            reset_button = st.button("Reset", key="reset_button", use_container_width = True)
    st.write("---")
    generate = st.button("Generate timestamps", key="generate_timestamp")
if st.session_state.include_eventtype:
    if change_button:
        st.session_state.event_type_map["LITE"] = new_lite
        st.session_state.event_type_map["VAR"] = new_var
        st.session_state.event_type_map["HADD"] = new_hadd
    if reset_button:
        st.session_state.event_type_map = DEFAULT_MAP
st.session_state
if generate:
    st.session_state.timeline = ""
    generate_timeline(username, password, date, hour, minutes, seconds, with_event_type, hindsight)
    st.session_state.show_timeline = True
if st.session_state.show_timeline and st.session_state.timeline:
    st.code(st.session_state.timeline, language="python")

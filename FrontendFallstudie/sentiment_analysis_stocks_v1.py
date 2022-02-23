# Import der benötigten Python Module und Bibliotheken
import csv
import json

import nltk
import numpy as np
import pandas as pd
from textblob import \
    TextBlob  # Modul für Sentiment Analyse für englische Tweets
from textblob_de import \
    TextBlobDE  # Modul für Sentiment Analyse für deutsche Tweets

# Import der in load_twitter_data geschriebene Funktion, um Tweets zu downloaden
from load_twitter_data_v1 import load_twitter_data

nltk.download('punkt') #beim ersten mal auskommentieren und mit ausführen, damit TextBlob richtig funktioniert :)


# Funktion , um die gedownloadeten Tweets in ein DataFrame zu laden
def load_data_in_dataframe(data_file):
    data = pd.read_csv(
        data_file,
        encoding='unicode_escape',
        on_bad_lines='skip',
        engine='python',
        delimiter='\n',
        names=['tweet'])  # lesen der .txt Datei mit den gedwonloadeten Tweets
    data = data.drop_duplicates()  # löschen aller doppelten Tweets
    return data


# Funktion, um den DataFrame von beschreibenden Tweets zu reinigen
def data_cleaning(data):
    data = data[data["tweet"].str.contains("https") ==
                False]  # löschen Tweets die Links enthalten
    data = data[
        data["tweet"].str.contains("#tradingview") ==
        False]  # löschen von Tweets, welche Aktienanalysen auf Tradingview beschreiben
    return data


# Funktion, um die Polarität jedes deutschen Tweets zu ermitteln
def sentiment_de(tweet):
    blob = TextBlobDE(tweet)  # nutzen von TextBlob für jeden deutschen Tweets
    return blob.sentiment.polarity  # zurükgeben der Polarität für jeden deutschen Tweet


# Funktion, um die Polarität jedes englischen Tweets zu ermitteln
def sentiment_en(tweet):
    blob = TextBlob(tweet)  # nutzen von TextBlob für jeden englischen Tweets
    return blob.sentiment.polarity  # zurückgeben der Polarität für jeden englischen Tweet


# Funktion, um die Tweets nach ihrer Polarität zu labeln
def label(sentiment):
    if float(sentiment) < 0:
        return "negative"  # Rückgabe des Labels "negativ" für Tweets mit Polarität kleiner 0
    elif float(sentiment) == 0:
        return "neutral"  # Rückgabe des Labels "neutral" für Tweets mit Polarität gleich 0
    elif float(sentiment) > 0:
        return "positive"  # Rückgabe des Labels "positiv" für Tweets mit Polarität größer 0


# Funktion, um die Sentiment Analyse auf die einzelnen deutschen Tweets aus dem DataFrame anzuwenden und das Label hinzuzufügen
def use_sentiment_analysis_de(data):
    data['Sentiment_TextBlob'] = data['tweet'].apply(
        sentiment_de
    )  # aufrufen der Funktion zum Anwenden von TextBlob auf die deutschen Tweets
    data["Sentiment_TextBlob_label"] = data["Sentiment_TextBlob"].apply(
        label
    )  # aufrufen der Funktion zum Hinzufügen der Labels für die deutschen Tweets
    return data


# Funktion, um die Sentiment Analyse auf die einzelnen englischen Tweets aus dem DataFrame anzuwenden und das Label hinzuzufügen
def use_sentiment_analysis_en(data):
    data['Sentiment_TextBlob'] = data['tweet'].apply(
        sentiment_en
    )  # aufrufen der Funktion zum Anwenden von TextBlob auf die englischen Tweets
    data["Sentiment_TextBlob_label"] = data["Sentiment_TextBlob"].apply(
        label
    )  # aufrufen der Funktion zum Hinzufügen der Labels für die englischen Tweets
    return data


# Funktion, um die neutralen Tweets aus dem Datensatz zu löschen
def clean_neutral(data):
    data = data[data["Sentiment_TextBlob_label"].str.contains("neutral") ==
                False]  # löschen der neutralen Tweets aus dem Datensatz
    return data


# Funktion, um den DataFrame so zu aggregieren, dass nur die Anzahl positiver, negativer und neutraler Tweets in absteigender Reihenfolge gezeigt wird
def sentiment_aggregated_output(data):
    data_sorted = data.groupby(
        by="Sentiment_TextBlob_label").count().sort_values(
            by='tweet', ascending=False
        )  # aggregieren und sortieren der Tweets, nach den Labels
    return data_sorted


# Funktion, um das Sentiment für das Frontend zu extrahieren
def sentiment_for_gui(data):
    return str(
        data.index[0])  # extrahieren des sentiments mit den meisten Tweets


# Funktion, um die zur Stimmung passenden Sektoren auszuwählen
def sector(sentiment):
    if sentiment == 'negative':
        return [
            'Healthcare', 'Utilities', 'Basic_Materials', 'Consumer_Defensive'
        ]  # Rückgabe der Sektoren für das Sentiment/Stimmung "negativ"
    if sentiment == 'neutral':
        return ['Industrials', 'Communication', 'Consumer_Cyclical', 'Energy'
                ]  # Rückgabe der Sektoren für das Sentiment/Stimmung "neutral"
    if sentiment == 'positive':
        return ['Financial_Services', 'Technology', 'Real_Estate'
                ]  # Rückgabe der Sektoren für das Sentiment/Stimmung "positiv"


# Funktion, um für jeden einzelnen Sektor Tweets runterzuladen und in einer .txt zu speichern
def sentiment_sector(sector_list):
    sector_positive_dict = {
    }  # erstellen des Dicts für die Sektoren und ihre Anzahl psoitiver Tweets
    for x in sector_list:
        try:
            file_name_sector = 'sector_tweet_' + x + '.txt'  # erstellen des Namens für die Datei für jeden Sektor
            search_query_sector = 'stocks ' + x + ' OR stock ' + x  # erstellen der search query für jeden Sektor, diesmal mit englischen Tweets
            load_twitter_data(
                file_name_sector, search_query_sector,
                2500)  # download von 2500 Tweets für jeden Sektor
            data_file_sector = 'sector_tweet_' + x + '.txt'
            tweets_sector = load_data_in_dataframe(
                data_file_sector)  # laden der Sektor Tweets in einen DataFrame
            tweets_sector = data_cleaning(
                tweets_sector)  # säubern der Sektor Tweets
            tweets_sector = use_sentiment_analysis_en(
                tweets_sector
            )  # nutzen der Sentiment Analyse für die Sektor Tweets
            tweets_sector = clean_neutral(
                tweets_sector)  # löschen der neutralen Sektor Tweets
            tweets_sector = sentiment_aggregated_output(
                tweets_sector)  # aggregieren der Sektor Tweets
            x = {
                x: tweets_sector.iloc[0][0]
            }  # extrahieren der Anzahl positiver Tweets
            sector_positive_dict.update(
                x
            )  # schreiben des Sektors und der zugehörigen Anzahl Tweets in ein Dict
        except:
            continue
    sector_positive_dict = sorted(
        sector_positive_dict.items(), key=lambda x: x[1], reverse=True
    )  # sortieren der Sektoren, nach der Anzahl der positiven Tweets
    return sector_positive_dict


# Funktion, um die Marktstimmung und die geordneten Sektoren in einer JSON zu speichern
def create_json_for_frontend(sentiment, sector_sentiment_dict):
    list_for_json = []  # erstellen der Liste um die JSON File zu füllen
    for x in sector_sentiment_dict:  # Schleife um die Liste zu befüllen
        list_for_json.append(
            x[0])  # hinzufügen der Sektoren aus dem Dict zu der Liste
    dict_frontend = {
        sentiment: list_for_json
    }  # erstellen eines neuen Dict für die JSON File, mit sentiment als key und den zugehörigen Sektoren sortiert als items
    with open('Data\json_for_frontend.json', 'w') as json_for_frontend:
        json.dump(dict_frontend,
                  json_for_frontend)  # schreiben des dicts in das JSON File


# Funktion, um alle Funktionen auf einmal ausführbar zu machen
def run_all_functions(file_name='test_run_all_v4.txt',
                      search_query='aktien OR aktie',
                      tweet_limit=10000):
    load_twitter_data(
        file_name, search_query, tweet_limit
    )  # aufrufen der Funktion, um Tweets von Twitter zu downloaden
    data_dataframe = load_data_in_dataframe(
        file_name
    )  # aufrufen der Funktion, um die Tweets in einen DataFrame zu laden
    cleaned_dataframe = data_cleaning(
        data_dataframe)  # aufrufen der Funktion, um den DataFrame zu säubern
    sentiment_dataframe = use_sentiment_analysis_de(
        cleaned_dataframe
    )  # aufrufen der Funktion, um die Sentiment Analyse auf die Tweets über Aktien anzuwenden
    #sentiment_dataframe = clean_neutral(sentiment_dataframe)
    aggregated_sentiment_dataframe = sentiment_aggregated_output(
        sentiment_dataframe)  # aufrufen der Funktion zum aggregiern der Tweets
    sentiment_for_frontend = sentiment_for_gui(
        aggregated_sentiment_dataframe
    )  # aufrufen der Funktion, um das im Frontend angezeigte Sentiment/Stimmung zu extrahieren
    print(sentiment_for_frontend)  # Ausgabe des extrahierten sentiments
    sector_for_sentiment = sector(
        sentiment_for_frontend
    )  # aufrufen der Funktion, um die Sektoren für das Frontend zu ermitteln
    print(sector_for_sentiment)  # Ausgabe der Sektoren
    sentiment_sector_list = sentiment_sector(
        sector_for_sentiment
    )  # aufrufen der Funktion, um für jedenn Sektor die Zustimmung zu ermitteln
    print(sentiment_sector_list)  # Ausgabe der Sektoren mit ihrer Zustimmung
    create_json_for_frontend(
        sentiment_for_frontend, sentiment_sector_list
    )  # aufrufen der Funktion zum erstellen der JSON Datei für die Übergabe an das Frontend


# file_name = 'test_run_all_v4.txt'
# search_query = 'aktien OR aktie '
# tweet_limit = 10000
# data = run_all_functions('test_run_all_v4.txt', 'aktien OR aktie ', 10000)
# print(data)

import tweepy  # import der Bibliothek um Tweets von Twitter zu downloaden

# set up des Clients um sich mit der Twitter API zu verbinden
client = tweepy.Client(
    bearer_token=
    'AAAAAAAAAAAAAAAAAAAAAGhzXwEAAAAA0RB69ciaU0PoDL8d7LQtQwac5cA%3D8JcZ8503QWHdc4mtRXgFAkEZ3CGlipbT6ZaEi21hf5GQZeXSxL'
)  # individueller Token einer APP auf der Twitter Entwicklungsseite, um Tweets downloaden zu können


# Funktion, um eine bestimmte Anzahl von Tweets runterzuladen und in einer .txt Datei zu speichern
def load_twitter_data(file_name, search_query, tweet_limit):
    with open(
            file_name, 'a+'
    ) as filehandle:  # erstellen und öffnen der Datei in der die Tweets abgelegt werden sollen, Name der Datei durch file_name einstellbar
        for tweet in tweepy.Paginator(
                client.search_recent_tweets,
                query=search_query,
                tweet_fields=['context_annotations', 'created_at'],
                max_results=100).flatten(limit=tweet_limit):
            try:
                filehandle.write('%s\n' % tweet.text)
            except:
                continue


# file_name = 'test_python.txt'
# search_query = 'aktien OR aktie gut '
# tweet_limit = 100

# load_twitter_data(file_name, search_query, tweet_limit)

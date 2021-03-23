## importing packages
###!pip install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint
## pip install twint
## pip install nest_asyncio
## pip install vaderSentiment
## pip install altair vega_datasets
## pip install vega
## alt.renderers.enable('notebook') if you work in Jupyter Notebook
## pip install spacy && python -m spacy download en

import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
import nltk
#nltk.download('vader_lexicon')
import altair as alt
from vega_datasets import data
import seaborn as sns
sns.set()
import spacy
nlp = spacy.load("en_core_web_sm")

## Only for scraping tweets
import twint
import nest_asyncio
nest_asyncio.apply()

## Phase One: collecting tweets from companies
## Note: it will save as .csv on local; if you repeat this, it will add to existing .csv file!

def scrape_tweets(bank, outfile, starting, ending):
    c = twint.Config()
    c.Search = bank  ## provide Twitter name
    c.Since = starting
    c.Until = ending
    c.Store_csv = True
    c.Output = outfile ## csv with bankname.csv
    twint.run.Search(c)

scrape_tweets('@chime', 'chime2.csv', '2020-08-02', '2021-03-21')
scrape_tweets('@RevolutApp', 'revolut2.csv', '2020-08-02', '2021-03-21')
scrape_tweets('@VaroBank', 'varo2.csv', '2020-08-02', '2021-03-21')
scrape_tweets('@Ally', 'ally2.csv', '2020-08-02', '2021-03-21')
scrape_tweets('@WellsFargo', 'wellsfargo2.csv', '2020-08-02', '2021-03-21')
scrape_tweets('@BankofAmerica', 'bankofamerica2.csv', '2020-08-02', '2021-03-21')
scrape_tweets('@Chase', 'chase2.csv', '2020-08-02', '2021-03-21')

########################################################
##########################################################
########################################################

date_cols=['date']

## Converting into pandas datrame for use
def make_pandas_dataframe(csvfilename):

    dataframe = pd.read_csv(csvfilename,  ## name of new pandas data frame;  name of csv
                usecols = ['date', 'time', 'username', 'tweet', 'language',
                           'mentions', 'likes_count'],
                            dtype={'time': 'str'},
                infer_datetime_format=True)
    return dataframe

chime2 = make_pandas_dataframe('chime2.csv')
revolut2 = make_pandas_dataframe('revolut2.csv')
varo2 = make_pandas_dataframe('varo2.csv')
ally2 = make_pandas_dataframe('ally2.csv')
wellsfargo2 = make_pandas_dataframe('wellsfargo2.csv')
bankofamerica2 = make_pandas_dataframe('bankofamerica2.csv')
chase2 = make_pandas_dataframe('chase2.csv')

## Making sure to get only tweets denoted by English ("en")
def filter_tweets_by_language(dataframe, language):
    dataframe_filtered = dataframe[dataframe.language.isin([language])]
    return dataframe_filtered

chime2 = filter_tweets_by_language(chime2, "en")
revolut2 = filter_tweets_by_language(revolut2, "en")
varo2 = filter_tweets_by_language(varo2, "en")
ally2 = filter_tweets_by_language(ally2, "en")
bankofamerica2 = filter_tweets_by_language(bankofamerica2, "en")
wellsfargo2 = filter_tweets_by_language(wellsfargo2, "en")
chase2 = filter_tweets_by_language(chase2, "en")

# Removing tweets authored by company itself or sponsored sports team
def filter_out_company(dataframe, company_handle):
    dataframe_filtered = dataframe[~dataframe['username'].isin([company_handle])]
    return dataframe_filtered    

chime2 = filter_out_company(chime2, 'chime')
chime2 = filter_out_company(chime2, 'dallasmavs')
revolut2 = filter_out_company(revolut2, 'revolutapp')
varo2 = filter_out_company(varo2, 'varobank')
ally2 = filter_out_company(ally2, 'ally')
wellsfargo2 = filter_out_company(wellsfargo2, 'wellsfargo')
wellsfargo2 = filter_out_company(wellsfargo2, 'unc_basketball')
bankofamerica2 = filter_out_company(bankofamerica2, 'bankofamerica')
bankofamerica2 = filter_out_company(bankofamerica2, 'panthers') 
chase2 = filter_out_company(chase2, 'chase')
chase2 = filter_out_company(chase2, 'warriors') 

# Removing tweets discussing sponsored sports team
def filter_out_company2(dataframe, team_name):
    dataframe_filtered = dataframe[~dataframe['tweet'].str.contains(team_name)]
    return dataframe_filtered

chime2 = filter_out_company2(chime2, '@dallasmavs')
bankofamerica2 = filter_out_company2(bankofamerica2, '@panthers')
wellsfargo2 = filter_out_company2(wellsfargo2, '@UNC_Basketball')
chase2 = filter_out_company2(chase2, '@warriors')

# Creating time markers for data transformation and graphing
def make_date_into_date_format(dataframe, companyname):
    dataframe['tweetdate'] = pd.to_datetime(dataframe['date'])
    dataframe['tweettime'] = pd.to_datetime(dataframe['time'])
    dataframe['tweetdayofweek'] = dataframe['tweetdate'].dt.dayofweek
    dataframe['tweetyr'] = dataframe['tweetdate'].dt.year
    dataframe['company'] = companyname
    return dataframe

chime2 = make_date_into_date_format(chime2, 'Chime')
revolut2 = make_date_into_date_format(revolut2, 'Revolut')
revolut2 = make_date_into_date_format(revolut2, 'Revolut')
ally2 = make_date_into_date_format(ally2, 'Ally Bank')
bankofamerica2 = make_date_into_date_format(bankofamerica2, 'Bank of America')
wellsfargo2 = make_date_into_date_format(wellsfargo2, 'Wells Fargo')
chase2 = make_date_into_date_format(chase2, 'Chase Bank')

########################################################
##########################################################
########################################################
#
# Conducting sentiment analysis on tweets
# Start with test - various phrases

from nltk.sentiment.vader import SentimentIntensityAnalyzer
vds = SentimentIntensityAnalyzer()

def testing_vader_polarity2(sample):
    sample2 = vds.polarity_scores(sample)
    print("The score for '{}' is: \n{}".format(sample, sample2))

testing_vader_polarity2("I went to the movies last week.")
testing_vader_polarity2("I went to the movie, yesterday. It was amazing! Everyone acted well!")
testing_vader_polarity2("I went to the movie last week. I didn't enjoy the acting or the subtitles.")
testing_vader_polarity2("I went to the movie last week. I really enjoyed the acting but not the subtitles.")
testing_vader_polarity2("I went to the movies last week. The subtitles were not useful.")
testing_vader_polarity2("I went to the movies last week. The acting was AWFUL!!")

# Calculating sentiment scores, and appending to current dataframes; 
# Also calculating mean scores by company
def make_vader_sentiment_scores(dataframe_vds_filter):
    dataframe_vds_filter['scores'] = dataframe_vds_filter['tweet'].apply(lambda tweet: vds.polarity_scores(tweet))
    dataframe_vds_filter['compound'] = dataframe_vds_filter['scores'].apply(lambda score_dict: score_dict['compound'])
    dataframe_vds_filter['compound*likes'] = dataframe_vds_filter['compound'] * (0.20 * (dataframe_vds_filter['likes_count'] + 5))
    return dataframe_vds_filter
chime2 = make_vader_sentiment_scores(chime2)
revolut2 = make_vader_sentiment_scores(revolut2)
varo2 = make_vader_sentiment_scores(varo2)
ally2 = make_vader_sentiment_scores(ally2)
bankofamerica2 = make_vader_sentiment_scores(bankofamerica2)
wellsfargo2 = make_vader_sentiment_scores(wellsfargo2)
chase2 = make_vader_sentiment_scores(chase2)

chime_sentiment_mean = chime2['compound'].mean()
revolut_sentiment_mean = revolut2['compound'].mean()
varo_sentiment_mean = varo2['compound'].mean()
ally_sentiment_mean = ally2['compound'].mean()
bankofamerica_sentiment_mean = bankofamerica2['compound'].mean()
wellsfargo_sentiment_mean = wellsfargo2['compound'].mean()
chase_sentiment_mean = chase2['compound'].mean()

print("The average compound sentiment score for Chime is: {}".format(chime_sentiment_mean))
print("The average compound sentiment score for Revolut is: {}".format(revolut_sentiment_mean))
print("The average compound sentiment score for Varo Bank is: {}".format(varo_sentiment_mean))
print("The average compound sentiment score for Ally Bank is: {}".format(ally_sentiment_mean))
print("The average compound sentiment score for Bank of America is: {}".format(bankofamerica_sentiment_mean))
print("The average compound sentiment score for Wells Fargo is: {}".format(wellsfargo_sentiment_mean))
print("The average compound sentiment score for Chase Bank is: {}".format(chase_sentiment_mean))


# Transforming data by week, and subsequent new dataframes for graphing
def new_weekly_dataframe(company, company_name):
    companyts = company.set_index('tweetdate')
    df_tweets = companyts.resample('W')['compound'].count().to_frame()
    df_sentiment = companyts.resample('W')['compound'].mean().to_frame()
    df_sentiment_likes = companyts.resample('W')['compound*likes'].mean().to_frame()
    df_tweets = df_tweets.rename(columns={'compound': 'tweet_count'}).reset_index()
    df_sentiment = df_sentiment.rename(columns={'compound': 'sentiment_avg'}).reset_index()
    df_sentiment_likes = df_sentiment_likes.rename(columns={'compound*likes': 'sentiment_avg+likes'}).reset_index()
    merge_table_1 = df_tweets.merge(df_sentiment,on='tweetdate') 
    merge_table_2 = merge_table_1.merge(df_sentiment_likes,on='tweetdate')
    merge_table_2['company'] = company_name
    return merge_table_2

chime_2 = new_weekly_dataframe(chime2, 'Chime')
revolut_2 = new_weekly_dataframe(revolut2, 'Revolut')
varo_2 = new_weekly_dataframe(varo2, 'Varo Bank')
ally_2 = new_weekly_dataframe(ally2, 'Ally Bank')
bankofamerica_2 = new_weekly_dataframe(bankofamerica2, 'Bank of America')
wellsfargo_2 = new_weekly_dataframe(wellsfargo2, 'Wells Fargo')
chase_2 = new_weekly_dataframe(chase2, 'Chase')

# Graphing Chime 
chime_weekly_tweets1 = alt.Chart(chime_2).mark_line().encode(
    alt.X('tweetdate:T', axis=alt.Axis(title='Week')),
    alt.Y('tweet_count', axis=alt.Axis(title='Number of tweets'))
)

chime_weekly_tweets1

chime_weekly_tweets2 = alt.Chart(chime_2).mark_line().encode(
    alt.X('tweetdate:T', axis=alt.Axis(title='Week')),
    alt.Y('sentiment_avg', scale=alt.Scale(domain=[-0.5,1]), axis=alt.Axis(title='Mean sentiment score'))
)

chime_weekly_tweets2

chime_weekly_tweets3 = alt.Chart(chime_2).mark_line().encode(
    alt.X('tweetdate:T', axis=alt.Axis(title='Week')),
    alt.Y('sentiment_avg+likes', scale=alt.Scale(domain=[-0.4,1]), axis=alt.Axis(title='Mean sentiment score + likes'))
)

chime_weekly_tweets3

####################################
#
# Creating dataset for graphing

new_data_1 = chime_2.append(revolut_2)
new_data_1 = new_data_1.append(varo_2)
new_data_1 = new_data_1.append(ally_2)
new_data_1 = new_data_1.append(bankofamerica_2)
new_data_1 = new_data_1.append(wellsfargo_2)
new_data_1 = new_data_1.append(chase_2)

# Creating altair charts - direct comparisons

alt.Chart(new_data_1).mark_bar().encode(
    x=alt.X('company', sort='-y'), 
    y='median(tweet_count)',
    color='company',
    strokeDash='company',
)

alt.Chart(new_data_1).mark_bar().encode(
    x=alt.X('company', sort='-y'), 
    y='median(sentiment_avg)',
    color='company',
    strokeDash='company',
)

alt.Chart(new_data_1).mark_bar().encode(
    x=alt.X('company', sort='-y'), 
    y='median(sentiment_avg+likes)',
    color='company',
    strokeDash='company',
)

# Preparing for charts - grouping by company to make charts
list(new_data_1.groupby('company')['tweet_count'].median().sort_values(ascending=False).index)

#################################33
# Side by side bar charts, ordered by median tweet count
company_order_by_tweets = list(new_data_1.groupby('company')['tweet_count'].median().sort_values(ascending=False).index)

base = alt.Chart(new_data_1).properties(width=550)

left = alt.Chart(new_data_1).mark_bar().encode(
    x=alt.X('median(tweet_count)',
           sort='descending',
##sort=alt.SortOrder('descending')
            title = 'Median weekly tweet count'), 
    y=alt.Y('company', 
            axis=None, 
           sort='-x'), ##axis=alt.Axis(orient='right')),  ## axis=alt.Axis(orient='right')
    color=alt.Color('company', legend=None),
    strokeDash='company',
).properties(width=300, height=250)


middle = base.encode(
    y=alt.Y('company', axis=None,
           sort=company_order_by_tweets),
    text=alt.Text('company'),
).mark_text(fontSize=18).properties(width=140, height=250)

right = alt.Chart(new_data_1).mark_bar().encode(
    x=alt.X('median(sentiment_avg)',
            scale=alt.Scale(domain=(0, 0.35)),
            title = 'Median weekly sentiment score'),
    y=alt.Y('company', axis=None,
           sort=company_order_by_tweets),
    color='company',
    strokeDash='company',
).properties(width=300, height=250)

alt.concat(left, middle, right, spacing=5)

###########################################
# 
# Side by side, weekly sentiment and median weekly tweet count
company_order_by_tweets = list(new_data_1.groupby('company')['tweet_count'].median().sort_values(ascending=False).index)

base = alt.Chart(new_data_1).properties(width=550)

left = alt.Chart(new_data_1).mark_bar().encode(
    x=alt.X('median(tweet_count)',
           sort='descending',
##sort=alt.SortOrder('descending')
            title = 'Median weekly tweet count'), 
    y=alt.Y('company', 
            axis=None, 
           sort='-x'), ##axis=alt.Axis(orient='right')),  ## axis=alt.Axis(orient='right')
    color=alt.Color('company', legend=None),
    strokeDash='company',
).properties(width=300, height=250)


middle = base.encode(
    y=alt.Y('company', axis=None,
           sort=company_order_by_tweets),
    text=alt.Text('company'),
).mark_text(fontSize=18).properties(width=140, height=250)

right = alt.Chart(new_data_1).mark_bar().encode(
    x=alt.X('median(sentiment_avg+likes)',
            scale=alt.Scale(domain=(0, 0.35)),
            title = 'Median weekly sentiment score (+ likes)'),
    y=alt.Y('company', axis=None,
           sort=company_order_by_tweets),
    color='company',
    strokeDash='company',
).properties(width=300, height=250)

alt.concat(left, middle, right, spacing=5)

###########################################
#
# Side by side comparing sentiment with sentiment+likes
base = alt.Chart(new_data_1).properties(width=550)

left = alt.Chart(new_data_1).mark_bar().encode(
    x=alt.X('median(sentiment_avg)',
           scale=alt.Scale(domain=(0, 0.35)),
           sort='descending',
##sort=alt.SortOrder('descending')
            title = 'Median weekly sentiment score'), 
    y=alt.Y('company', 
            axis=None),
    color=alt.Color('company', legend=None),
    strokeDash='company',
).properties(width=300, height=250)


middle = base.encode(
    y=alt.Y('company', axis=None),
    text=alt.Text('company'),
).mark_text(fontSize=18).properties(width=140, height=250)

right = alt.Chart(new_data_1).mark_bar().encode(
    x=alt.X('median(sentiment_avg+likes)',
            scale=alt.Scale(domain=(0, 0.35)),
            title = 'Median weekly sentiment score (+ likes)'),
    y=alt.Y('company', axis=None),
    color='company',
    strokeDash='company',
).properties(width=300, height=250)

alt.concat(left, middle, right, spacing=5)

##    alt.Y('sentiment_avg+likes', scale=alt.Scale(domain=[-0.4,1]), axis=alt.Axis(title='Mean sentiment score + likes'))

#####################################3
#
# Weekly sentiment average, using tooltip to hover with pointer

highlight = alt.selection(type='single', on='mouseover',
                          fields=['company'], nearest=False, empty="none")

## Nearest=True should keep color until moved

alt.Chart(new_data_1).mark_line().encode(
    x='tweetdate',
    y='sentiment_avg',
    color=alt.condition(highlight, 'company', alt.value("oldlace")),
    tooltip=['company', "sentiment_avg", "tweetdate"]
).add_selection(
    highlight
)

######################################3
#
# Weekly tweet count, click legend to see individual

source = new_data_1
s2 = pd.DataFrame(source, columns=['company'])
selection = alt.selection_multi(on='mouseover', fields=['company'], nearest=True)

chart = alt.Chart(new_data_1).mark_line().encode(
    alt.X('tweetdate', axis=alt.Axis(domain=False), title='Week of tweets'),
    alt.Y('tweet_count', title='Average weekly tweet count'),
    alt.Color('company'),
    opacity=alt.condition(selection, alt.value(1), alt.value(.075)),
    tooltip=['company', "sentiment_avg", "tweetdate"])

hover_legend = alt.Chart(s2).mark_circle(size=100).encode(
     alt.Y('company', axis=alt.Axis(orient='right', domain=False, ticks=False), title=None),
     alt.Color('company', legend=None),
     opacity=alt.condition(selection, alt.value(1), alt.value(.075))
).add_selection(selection)
          
(chart | hover_legend).configure_view(strokeWidth=0)

##########################
#
# Weekly sentiment score, click on legend

source = new_data_1
s2 = pd.DataFrame(source, columns=['company'])
selection = alt.selection_multi(on='mouseover', fields=['company'], nearest=True)

chart = alt.Chart(new_data_1).mark_line().encode(
    alt.X('tweetdate', axis=alt.Axis(domain=False), title='Week of tweets'),
    alt.Y('sentiment_avg', title='Average sentiment score'),
    alt.Color('company'),
    opacity=alt.condition(selection, alt.value(1), alt.value(.075)),
    tooltip=['company', "sentiment_avg", "tweetdate"])

hover_legend = alt.Chart(s2).mark_circle(size=100).encode(
     alt.Y('company', axis=alt.Axis(orient='right', domain=False, ticks=False), title=None),
     alt.Color('company', legend=None),
     opacity=alt.condition(selection, alt.value(1), alt.value(.075))
).add_selection(selection)
          
(chart | hover_legend).configure_view(strokeWidth=0)

#################################
#
# Average sentiment score + likes, touch legend symbol

source = new_data_1
s2 = pd.DataFrame(source, columns=['company'])
selection = alt.selection_multi(on='mouseover', fields=['company'], nearest=True)

chart = alt.Chart(new_data_1).mark_line().encode(
    alt.X('tweetdate', axis=alt.Axis(domain=False), title='Week of tweets'),
    alt.Y('sentiment_avg+likes', title='Average sentiment score, weight by likes'),
    alt.Color('company'),
    opacity=alt.condition(selection, alt.value(1), alt.value(.075)),
    tooltip=['company', "sentiment_avg+likes", "tweetdate"])

hover_legend = alt.Chart(s2).mark_circle(size=100).encode(
     alt.Y('company', axis=alt.Axis(orient='right', domain=False, ticks=False), title=None),
     alt.Color('company', legend=None),
     opacity=alt.condition(selection, alt.value(1), alt.value(.075))
).add_selection(selection)
          
(chart | hover_legend).configure_view(strokeWidth=0)

########################################################
##########################################################
########################################################
#
# Preparations for reporting unigrams and igrams

from sklearn.feature_extraction.text import CountVectorizer

bag_of_words_vectorizer = CountVectorizer()

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

################################
#
# Calculating unigrams

def process_my_data_for_unigrams(unigrams_data, company_name):
    ng_counter=CountVectorizer(max_features=300, 
                               ngram_range=(1,1), 
                               stop_words=STOP_WORDS.union({company_name, 
                                                            'https', 'co', 'll', 've', 'amp',
                                                            'money', 'account', 'bank', 'card', 'deposit', 'check'}))
    ng_counter=ng_counter.fit(unigrams_data['tweet'])
    unigram_counts = ng_counter.transform(unigrams_data['tweet'])
    unigram_counts.toarray()
    unigram_counts.sum(axis=0)
    unigram_counts_total = unigram_counts.sum(axis=0)
    words_1 = unigram_counts_total.argsort()[0,-50:]
    words_1_array = np.squeeze(np.asarray(words_1))
    company_unigram_phrase = []
    company_unigram_count = []
    for number in words_1_array[::-1]:
        company_unigram_phrase.append(ng_counter.get_feature_names()[number])
        company_unigram_count.append(unigram_counts_total[0,number])
    company_unigrams = pd.DataFrame({'phrase': company_unigram_phrase, 'count': company_unigram_count})
    return company_unigrams
    
chime_unigrams = process_my_data_for_unigrams(chime2, 'chime')
revolut_unigrams = process_my_data_for_unigrams(revolut2, 'revolutapp')
varo_unigrams = process_my_data_for_unigrams(varo2, 'varobank')
ally_unigrams = process_my_data_for_unigrams(ally2, 'ally')
bankofamerica_unigrams = process_my_data_for_unigrams(bankofamerica2, 'bankofamerica')
wellsfargo_unigrams = process_my_data_for_unigrams(wellsfargo2, 'wellsfargo')
chase_unigrams = process_my_data_for_unigrams(chase2, 'chase')


###############################################
#
# Charting unigrams by company

def unigram_chart(company_unigram):
    bars = alt.Chart(company_unigram).mark_bar().encode(
        x='count:Q',
        y=alt.Y("phrase:O", sort ='-x')
    )

    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text='count:Q'
    )

    return (bars + text).properties(height=900)
    
unigram_chart(chime_unigrams)
unigram_chart(revolut_unigrams)
unigram_chart(varo_unigrams)
unigram_chart(ally_unigrams)
unigram_chart(bankofamerica_unigrams)
unigram_chart(wellsfargo_unigrams)
unigram_chart(chase_unigrams)

#########################################
#
# Charting bigrams

def process_my_data_for_bigrams(bigrams_data, company_name):
    ng_counter=CountVectorizer(max_features=300, 
                               ngram_range=(2,2),
                               stop_words=STOP_WORDS.union({company_name, 'https', 'co', 'll', 've'}))
    ng_counter=ng_counter.fit(bigrams_data['tweet'])
    bigram_counts = ng_counter.transform(bigrams_data['tweet'])
    bigram_counts.toarray()
    bigram_counts.sum(axis=0)
    bigram_counts_total = bigram_counts.sum(axis=0)
    words_1 = bigram_counts_total.argsort()[0,-50:]
    words_1_array = np.squeeze(np.asarray(words_1))
    company_bigram_phrase = []
    company_bigram_count = []
    for number in words_1_array[::-1]:
        company_bigram_phrase.append(ng_counter.get_feature_names()[number])
        company_bigram_count.append(bigram_counts_total[0,number])
    company_bigrams = pd.DataFrame({'phrase': company_bigram_phrase, 'count': company_bigram_count})
    return company_bigrams
    
chime_bigrams = process_my_data_for_bigrams(chime2, 'chime')
revolut_bigrams = process_my_data_for_bigrams(revolut2, 'revolutapp')
varo_bigrams = process_my_data_for_bigrams(varo2, 'varobank')
ally_bigrams = process_my_data_for_bigrams(ally2, 'allycare')
bankofamerica_bigrams = process_my_data_for_bigrams(bankofamerica2, 'bankofamerica')
wellsfargo_bigrams = process_my_data_for_bigrams(wellsfargo2, 'wells')
chase_bigrams = process_my_data_for_bigrams(chase2, 'chase')

def bigram_chart(company_bigram):
    bars = alt.Chart(company_bigram).mark_bar().encode(
        x='count:Q',
        y=alt.Y("phrase:O", sort ='-x')
    )

    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text='count:Q'
    )

    return (bars + text).properties(height=900)
    
bigram_chart(chime_bigrams)
bigram_chart(revolut_bigrams)
bigram_chart(varo_bigrams)
bigram_chart(ally_bigrams)
bigram_chart(bankofamerica_bigrams)
bigram_chart(wellsfargo_bigrams)
bigram_chart(chase_bigrams)


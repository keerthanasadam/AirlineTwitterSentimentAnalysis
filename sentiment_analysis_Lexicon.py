import pandas as pd
import re

POSITIVEWORDLIST_FILENAME = "positive-words.txt"
STOPWORDLIST_FILENAME="stopwords.txt"
NEGATIVEWORDLIST_FILENAME = "negative-words.txt"
def loadWords(FILENAME):
    """
    Returns a list of positive and negative  words. Words are strings of lowercase letters.
    """
    inFile = open(FILENAME, 'r')
    wordList = []
    for line in inFile:
        if not line.lstrip().startswith(';'):
            wordList.append(line.strip().lower())
    return wordList

def getStopWords(Filename):
    """
    Returns list of stopwords.
    """
    stopwords=[]
    stop_file=open(Filename,"r");
    for line in stop_file:
        stopwords.append(line.strip());
    return stopwords
    
def readcsv():
    '''
    read specific columns from csv file
    '''
    
    tweets=pd.read_csv('Tweets.csv')
    df=tweets.copy()[['airline_sentiment','text']]
    tweets_pos=df.copy()[df.airline_sentiment=='positive'][:]
    tweets_neg=df.copy()[df.airline_sentiment=='negative'][:]
    tweets_text=tweets_pos.text.tolist()+tweets_neg.text.tolist()
    tweets_sentiment=tweets_pos.airline_sentiment.tolist()+tweets_neg.airline_sentiment.tolist() 
    preprocessed_tweet=preprocess(tweets_text)
    tweet_dict=dict(zip(preprocessed_tweet,tweets_sentiment))  
    return tweet_dict
    
def preprocess(tweets_text):
    '''
    Performing preprocessing for each tweet text and returns Preprocessed tweet
    '''
    processed_text=[]
    for text in tweets_text:
        #remove weblinks from a string
        text=re.sub("https?:\/\/.*[\r\n]*", '', text)
        #remove airline name        
        text=re.sub("@[a-zA-Z]*","",text)
        #replacing non-aplhabetic characters with space
        text=re.sub("\W"," ",text)
        #remove numbers
        text=re.sub("[0-9]*\.?[0-9]*","",text)
         # replace hashtag word with word        
        text=re.sub(r"#[^\s]",r"\1",text);
        #replacing multiple spaces with single space
        text=re.sub("\s{2,}"," ",text)
        #performing stemming
        #text=' '.join([PorterStemmer().stem_word(word) for word in text.split(" ")]) 
        text=' '.join(word for word in text.split() if word not in stopword_list)
        processed_text.append(text.lower().strip())
    return processed_text

def score_calculate(tweets_dict):
    '''
    Updating sentiment of tweet
    '''
    dict_update=tweets_dict.copy()
    pos_count=+1
    neg_count=+1
    for k,v in dict_update.items():
        tokens=k.split(' ')
        for i in tokens:
            if i in PositiveList:
                pos_count=pos_count+1
            if i in NegativeList:
                neg_count=neg_count+1
        total_count=pos_count-neg_count
        if total_count>0:
            dict_update[k]='positive'
        else:
            dict_update[k]='negative'
    return dict_update
    
def accuracy(old_dict,new_dict):
    '''
    Returns accuracy  
    '''
    assert len(old_dict)==len(new_dict), " old and updated dict don't have same no of values"
    total_prediction=0
    correct_prediction=0
    for key_old in old_dict.keys():
        if key_old in new_dict.keys():
            if(old_dict[key_old]==new_dict[key_old]):
                correct_prediction=correct_prediction+1
            total_prediction=total_prediction+1
    return (correct_prediction/float(total_prediction))*100
    
if __name__ == '__main__':
    PositiveList = loadWords(POSITIVEWORDLIST_FILENAME)
    NegativeList =loadWords(NEGATIVEWORDLIST_FILENAME)
    stopword_list=getStopWords(STOPWORDLIST_FILENAME);
    tweet_dict=readcsv()
    dict_update=score_calculate(tweet_dict)
    accuracy_value=accuracy(tweet_dict,dict_update)
    print("Accuracy using Lexicon approach using 2 classes:",accuracy_value)
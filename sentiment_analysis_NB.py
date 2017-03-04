
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from collections import defaultdict
from collections import Counter

STOPWORDLIST_FILENAME="stopwords.txt"
Positive_List=defaultdict(int)
Negative_List=defaultdict(int)
Neutral_List=defaultdict(int)
all_List=defaultdict(int)
def getStopWords(Filename):
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
    X_train, X_test, y_train, y_test= train_test_split(preprocessed_tweet,tweets_sentiment, test_size=0.2, random_state=0)
    all_count=len(X_train)
    tweet_dict_train=dict(zip(X_train,y_train))
    tweet_dict_test=dict(zip(X_test,y_test))
    (pos_count,neg_count)=featurelist(tweet_dict_train)  
    return (tweet_dict_train,tweet_dict_test,all_count,pos_count,neg_count)    

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
        text=re.sub("[0-9]*\.?[0-9]*","",text)
        # replace hashtag word with word        
        text=re.sub(r"#[^\s]",r"\1",text);
        #replacing multiple spaces with single space
        text=re.sub("\s{2,}"," ",text)
        #performing stemming
        #text=' '.join([PorterStemmer().stem_word(word) for word in text.split(" ")]) 
        text=' '.join(word for word in text.split() if word not in stopword_list)
        processed_text.append(text.lower().strip())
    #print(processed_text)
    return processed_text


def featurelist(tweets_dict):
     pos_text=0
     neg_text=0
     for k,v in tweets_dict.items():
        if v=='positive':
            pos_text=pos_text+1
        else:
            neg_text+=1
        tokens=k.split()
        for ele in tokens:
                all_List[ele]+=1
                if v=='positive':
                    Positive_List[ele]+=1
                else:
                    Negative_List[ele]+=1
     return (pos_text,neg_text)
def score_calculate(tweet,model,all_count,sen_count):
    '''
    Updating sentiment of tweet
    '''
    unigram=model  
    prior_prob=sen_count/all_count
    vocab_size=len(Counter(all_List))
    #print(vocab_size)
    size=sum(unigram.values())
    tokens=tweet.split(' ')
    likelihood_prob=1.0
    for i in range(len(tokens)):
            likelihood_prob*=float(unigram[tokens[i]]+1)/(size+vocab_size)
    prob_value=prior_prob*likelihood_prob
    return prob_value

def predict(tweets_dict,pos_model,neg_model):
    dict_update=tweets_dict.copy()
    pos_prob=0.0
    neg_prob=0.0
    for k,v in dict_update.items():
        pos_prob=score_calculate(k,pos_model,all_count,pos_count)
        neg_prob=score_calculate(k,neg_model,all_count,neg_count)
        if(pos_prob>neg_prob):
            dict_update[k]='positive'
        else:
            dict_update[k]='negative'
            
    return dict_update               

def accuracy(old_dict,new_dict):
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
    pos_text=0
    neg_text=0
    stopword_list=getStopWords(STOPWORDLIST_FILENAME);
    (tweet_dict_train,tweet_dict_test,all_count,pos_count,neg_count)=readcsv() 
    dict_update=predict(tweet_dict_test,Positive_List,Negative_List)
    accuracy_value=accuracy(tweet_dict_test,dict_update)
    print("Accuracy using Naive Bayes approach using 2 classes:",accuracy_value)
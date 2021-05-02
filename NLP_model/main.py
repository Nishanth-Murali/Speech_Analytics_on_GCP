import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re
import string
import time

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import urllib.request
from random import random
from random import randrange

import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pickle

from wordcloud import WordCloud
from textblob import TextBlob
import numpy as np
import math
import nltk
from nltk import word_tokenize, pos_tag
from gensim import matutils, models
import scipy.sparse
from subscribe import subscribe_class

from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
from google.cloud import storage
import os

from speech_to_text import speech_test
from flask import Flask


st = speech_test()

file_count = random()

# TODO(developer)
# project_id = "your-project-id"
# subscription_id = "your-subscription-id"
# Number of seconds the subscriber should listen for messages
#timeout = 30.0


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credential_file.json'


subscriber = pubsub_v1.SubscriberClient()
# The `subscription_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/subscriptions/{subscription_id}`
subscription_path = subscriber.subscription_path('genial-current-311020', 'sub_url') # pylint: disable=no-member

# create storage client
storage_client = storage.Client.from_service_account_json('./credential_file.json')
# get bucket with name
bucket = storage_client.get_bucket('transcript-input-bucket-6344')


app = Flask(__name__)
text_str_list = []

flag = True


@app.route('/')
def model():
    global text_str_list

    def run_model(text_str):

        def clean_text_round1(text_str):
            '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
            text_str = text_str.lower()
            text_str = re.sub(r'\[.*?\]', '', text_str)
            text_str = re.sub(r'[%s]' % re.escape(string.punctuation), '', text_str)
            text_str = re.sub(r'\w*\d\w*', '', text_str)
            return text_str

        # round1 = lambda x: clean_text_round1(x)


        # In[30]:
        sample_t1 = text_str

        sample_text_clean1 = clean_text_round1(sample_t1)
        print(sample_text_clean1)


        # In[31]:


        # Apply a second round of cleaning
        def clean_text_round2(text_str):
            '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
            text_str = re.sub(r'[‘’“”…]', '', text_str)
            text_str = re.sub(r'\n', '', text_str)
            return text_str

        # round2 = lambda x: clean_text_round2(x)


        # In[32]:


        sample_text_clean2 = clean_text_round2(sample_text_clean1)
        print(sample_text_clean2)


        # In[37]:


        corpus = pd.DataFrame([sample_text_clean2])


        # In[38]:


        # Let's pickle it for later use
        # data_df.to_pickle("corpus.pkl")


        # In[40]:


        data_clean = pd.DataFrame([sample_text_clean2])


        # In[41]:



        cv = CountVectorizer(stop_words='english')
        data_cv = cv.fit_transform([sample_text_clean2])
        data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
        data_dtm.index = data_clean.index
        data_dtm


        # In[44]:


        # Let's pickle it for later use
        data_dtm.to_pickle("dtm.pkl")

        # Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
        data_clean.to_pickle('data_clean.pkl')
        pickle.dump(cv, open("cv.pkl", "wb"))


        # # Part-2

        # In[45]:


        # Read in the document-term matrix


        data = data_dtm
        data = data_clean.transpose()


        # In[46]:


        # Find the top 30 words said by each comedian
        top_dict = {}
        for c in data.columns:
            top = data[c].sort_values(ascending=False).head(30)
            top_dict[c]= list(zip(top.index, top.values))


        # In[47]:


        # Print the top 15 words said by each speaker
        #for speaker, top_words in top_dict.items():
            #print(speaker)
            #print(', '.join([word for word, count in top_words[0:14]]))
            #print('---')


        # In[48]:


        # Look at the most common top words --> add them to the stop word list


        # Let's first pull out the top 30 words for each comedian
        words = []
        for comedian in data.columns:
            top = [word for (word, count) in top_dict[comedian]]
            for t in top:
                words.append(t)


        # In[49]:


        # Let's aggregate this list and identify the most common words along with how many routines they occur in
        Counter(words).most_common()


        # In[50]:


        add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]


        # In[51]:


        # Let's update our document-term matrix with the new list of stop words

        # Read in cleaned data
        data_clean = pd.read_pickle('data_clean.pkl')

        # Add new stop words
        stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

        # Recreate document-term matrix
        cv = CountVectorizer(stop_words=stop_words)
        data_cv = cv.fit_transform([data_clean[0][0]])
        data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
        data_stop.index = data_clean.index

        # Pickle it for later use

        pickle.dump(cv, open("cv_stop.pkl", "wb"))
        data_stop.to_pickle("dtm_stop.pkl")


        # In[53]:


        # Let's make some word clouds!
        # Terminal / Anaconda Prompt: conda install -c conda-forge wordcloud


        wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
                    max_font_size=150, random_state=42).generate(data_clean[0][0])


        # In[54]:



        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wc)
        plt.axis("off")
        plt.tight_layout(pad = 0)

        up_file_name = 'folder' + str(file_count) + '1.jpg'
        plt.savefig(up_file_name)
        st.upload_blob('image-output-bucket-6344', up_file_name, 'folder' + str(file_count) + '/' + up_file_name)


        # # Part-3

        # # Sentiment analysis

        # In[55]:


        # We'll start by reading in the corpus, which preserves word order


        data = corpus


        # In[56]:


        # Create quick lambda functions to find the polarity and subjectivity of each routine
        # Terminal / Anaconda Navigator: conda install -c conda-forge textblob


        pol = lambda x: TextBlob(x).sentiment.polarity
        sub = lambda x: TextBlob(x).sentiment.subjectivity

        data['polarity'] = data[0].apply(pol)
        data['subjectivity'] = data[0].apply(sub)


        # In[57]:


        # Let's plot the results


        plt.rcParams['figure.figsize'] = [12, 8]


        x = data.polarity.iloc[0]
        y = data.subjectivity.iloc[0]
        plt.scatter(x, y, color='blue')
        plt.text(x-.015, y+.001, '(Polarity, Subjectivity)', fontsize=10)
        plt.xlim(-.01, .12) 
            
        plt.title('Sentiment Analysis', fontsize=20)
        plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
        plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

        up_file_name = 'folder' + str(file_count) + '2.jpg'
        plt.savefig(up_file_name)
        st.upload_blob('image-output-bucket-6344', up_file_name, 'folder' + str(file_count) + '/' + up_file_name)


        # # Sentiment over time

        # In[58]:


        # Split each routine into 10 parts


        def split_text(text, n=10):
            '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

            # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
            length = len(text)
            size = math.floor(length / (n + 1) )
            start = np.arange(0, length+1, size+1) + 1
            
            # Pull out equally sized pieces of text and put it into a list
            split_list = []
            for piece in range(n):
                split_list.append(text[start[piece]:start[piece]+size])
            return split_list


        # In[59]:


        # Let's create a list to hold all of the pieces of text
        list_pieces = []
        split = split_text(data[0][0])
        list_pieces.append(split)


        # In[60]:


        # Calculate the polarity for each piece of text

        polarity_transcript = []
        for lp in list_pieces:
            polarity_piece = []
            for p in lp:
                polarity_piece.append(TextBlob(p).sentiment.polarity)
            polarity_transcript.append(polarity_piece)


        # In[61]:


        #Show the plot for the speaker
        plt.plot(polarity_transcript[0])
        plt.plot(np.arange(0,10), np.zeros(10))
        plt.title('Sentiment over time')
        up_file_name = 'folder' + str(file_count) + '3.jpg'
        plt.savefig(up_file_name, bbox_inches='tight')
        st.upload_blob('image-output-bucket-6344', up_file_name, 'folder' + str(file_count) + '/' + up_file_name)


        # # Part-4

        # In[62]:


        # Let's read in our document-term matrix


        data = data_stop


        # In[63]:


        # Import the necessary modules for LDA with gensim
        # Terminal / Anaconda Navigator: conda install -c conda-forge gensim



        # In[64]:


        # One of the required inputs is a term-document matrix
        tdm = data.transpose()


        # In[65]:


        # We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
        sparse_counts = scipy.sparse.csr_matrix(tdm)
        corpus = matutils.Sparse2Corpus(sparse_counts)


        # In[66]:


        # Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
        cv = pickle.load(open("cv_stop.pkl", "rb"))
        #id2word = dict((v, k) for k, v in cv.vocabulary_.items())


        # In[67]:




        def nouns(text):
            '''Given a string of text, tokenize the text and pull out only the nouns.'''
            is_noun = lambda pos: pos[:2] == 'NN'
            tokenized = word_tokenize(text)
            all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
            return ' '.join(all_nouns)


        # In[68]:


        # Read in the cleaned data, before the CountVectorizer step
        data_clean = pd.read_pickle('data_clean.pkl')


        # In[69]:


        # Apply the nouns function to the transcripts to filter only on nouns
        data_nouns = pd.DataFrame([nouns(data_clean[0][0])])


        # In[70]:


        # Create a new document-term matrix using only nouns


        # Re-add the additional stop words since we are recreating the document-term matrix
        add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                        'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
        stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

        # Recreate a document-term matrix with only nouns
        cvn = CountVectorizer(stop_words=stop_words)
        data_cvn = cvn.fit_transform([data_nouns[0][0]])
        data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
        data_dtmn.index = data_nouns.index
        # data_dtmn


        # In[71]:


        # Create the gensim corpus
        corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

        # Create the vocabulary dictionary
        id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())


        # In[72]:


        # Let's try 4 topics
        #ldan = models.LdaModel(corpus=corpusn, num_topics=4, id2word=id2wordn, passes=10)


        # In[73]:


        # Let's create a function to pull out nouns from a string of text
        def nouns_adj(text):
            '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
            is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
            tokenized = word_tokenize(text)
            nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
            return ' '.join(nouns_adj)


        # In[74]:


        # Apply the nouns function to the transcripts to filter  on nouns + adjective
        data_nouns_adj = pd.DataFrame([nouns_adj(data_clean[0][0])])


        # In[75]:


        # Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df
        cvna = CountVectorizer(stop_words=stop_words, max_df=1)
        data_cvna = cvna.fit_transform([data_nouns_adj[0][0]])
        data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
        data_dtmna.index = data_nouns_adj.index


        # In[76]:


        # Create the gensim corpus
        corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

        # Create the vocabulary dictionary
        id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())


        # In[77]:


        # Our final LDA model 
        ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=80)


        # In[78]:


        def getSize(txt, font):
            testImg = Image.new('RGB', (1, 1))
            testDraw = ImageDraw.Draw(testImg)
            return testDraw.textsize(txt, font)

        def topic_to_img(text):
            global file_count
            #file_count = random()
            #fontname = "C:/Windows/Fonts/Arial.ttf"
            #fontsize = 24 
            text = text
            
            colorText = "black"
            colorOutline = "red"
            colorBackground = "white"


            font = ImageFont.load_default()
            width, height = getSize(text, font)
            img = Image.new('RGB', (width+4, height+4), colorBackground)
            d = ImageDraw.Draw(img)
            d.text((2, height/2), text, fill=colorText, font=font)
            d.rectangle((0, 0, width+3, height+3), outline=colorOutline)

            up_file_name = 'folder' + str(file_count) + '4.jpg'
            img.save(up_file_name)
            st.upload_blob('image-output-bucket-6344', up_file_name, 'folder' + str(file_count) + '/' + up_file_name)
            
            # img.save("C:/Users/nisha/Pictures/image20.png")
            # img.show()


        # Let's take a look at which topics each transcript contains
        corpus_transformed = ldana[corpusna]
        topic_dict = {0: 'mom, parents', 1: 'husband, wife', 2: 'guns', 3: 'profanity'}
        final_topic_list = list(zip([a for [(a,b)] in corpus_transformed], data_dtmna.index))
        topic_to_img(topic_dict[final_topic_list[0][0]])
        
    def callback(message):
        global text_str_list
        # blob.make_public()
        # url = blob.public_url
        # get bucket data as blob
        blob = bucket.get_blob(message.data.decode('utf-8'))
        # convert to string
        # json_data = blob.download_as_string()
        # url = message.data.decode('utf-8')
        # url = str(message.data)
        text_str_list.append(blob.download_as_string().decode('utf-8'))
        print(text_str_list)
        print('-------------------')
        #run_model(text_str)
        message.ack()
        # return msg

    

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}..\n")

    

    # Wrap subscriber in a 'with' block to automatically call close() when done.
    #with subscriber:
        #global timeout
    try:
        # When `timeout` is not set, result() will block indefinitely,
        # unless an exception is encountered first.
        streaming_pull_future.result(timeout=30)
    except TimeoutError:
        streaming_pull_future.cancel()


    for i in range(len(text_str_list)):
        print(i)
        run_model(text_str_list[i])


    # class subscribe_class:


    


    # sc = subscribe_class()

    # sample_text = str('Intro\nFade the music out. Let’s roll. Hold there. Lights. Do the lights. Thank you. Thank you very much. I appreciate that. I don’t necessarily agree with you, but I appreciate very much. Well, this is a nice place. This is easily the nicest place For many miles in every direction. That’s how you compliment a building And shit on a town with one sentence. It is odd around here, as I was driving here. There doesn’t seem to be any difference Between the sidewalk and the street for pedestrians here. People just kind of walk in the middle of the road. I love traveling And seeing all the different parts of the country. I live in New York. I live in a– There’s no value to your doing that at all.','“The Old Lady And The Dog”\nI live– I live in New York. I always– Like, there’s this old lady in my neighborhood, And she’s always walking her dog. She’s always just– she’s very old. She just stands there just being old, And the dog just fights gravity every day, just– The two of them, it’s really– The dog’s got a cloudy eye, and she’s got a cloudy eye, And they just stand there looking at the street In two dimensions together, and– And she’s always wearing, like, this old sweater dress. I guess it was a sweater when she was, like, 5’10”, But now it’s just, like, this sweater And her legs are– her legs are a nightmare. They’re just white with green streaks and bones sticking out. Her legs are awful. I saw a guy with no legs wheeling by, And he was like, “yecch, no thank you. “I do not want those. “I’d rather just have air down here like I have Than to look down at that shit.” I see these two all the time, and I always look at them, And I always think, “god, I hope she dies first.” I do. I hope she dies first, for her sake, Because I don’t want her to lose the dog. I don’t think she’ll be able to handle it. If she dies– If the old lady dies first, I’m not worried about the dog Because the dog doesn’t even know about the old lady. This dog is aware of three inches around his head. He’s living in two-second increments. The second he’s in and the one he just left Is all he knows about, But if he dies, this lady, she’s gonna be destroyed Because this dog is all she has, And I know he’s all she has because she has him. There’s no– If she had one person in her life, She would not keep this piece of shit little dog. Even if just some young woman in her building one morning Were to say, “good morning, gladys,” She’d be like, “good,” And just flush him down the toilet, just– Poom! Poom! The dog just keeps bumping on the drain. Poom! “” she gives up. Ends up just shitting on her dog for the rest of her life. P-p-p! Poom!')
    # sample_text = sc.get_msg_from_sub()

    # Apply a first round of text cleaning techniques


    return 'OK'


if __name__ == '__main__':
    print('listener activated')
    model()
    app.run()

    

    

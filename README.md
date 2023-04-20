## Text Classifier for Identifying Toxic Comments on Wikipedia Talk Edit Pages

**Introduction:**

Wikipedia is the largest and most popular reference work on the internet with millions of contributors who can make edits to pages. The Talk edit pages serve as the key community interaction forum where the contributing community discusses or debates changes pertaining to a particular topic. This project aims to build a predictive model using NLP and machine learning to identify toxic comments in the Talk edit pages on Wikipedia.

**Objective:**

The objective of this project is to develop a text classification model using NLP text processing and the Support Vector Classifier (SVC) model in Python to determine toxic comments in the Talk edit pages on Wikipedia, and then identify the top terms from the toxic comments.

**Step-by-process:**

1. **Loaded the Wikipedia comments dataset acquired from Kaggle:** The dataset was loaded using the pandas' library. It contained identifiers, comment text, and labels for toxicity (0 for non-toxic, 1 for toxic).

2. **Preprocessed the data:** The raw text data was preprocessed to make it suitable for analysis. This involved several steps:

    a. **Text cleanup:** Removed IP addresses and URLs using regular expressions.

    b. **Normalization:** Converted all text to lowercase to standardize the text data.

    c. **Tokenization:** Split the text into individual words using the word_tokenize function from the NLTK library.

    d. **Removed stop words and punctuation:** Removed common stop words (e.g., 'the', 'and', 'is') and punctuation marks from the text data. Also, removed contextual stop words that may be specific to Wikipedia (e.g., 'article', 'page', 'talk').

3. **Split the dataset into train and test sets:** Separated the dataset into a 70:30 train-test split using the train_test_split function from the sklearn library.

4. **Vectorized the text data using TF-IDF:** Converted the cleaned and tokenized text data into a vector space model using the Tf-idf-Vectorizer from sklearn. This allowed the text data to be processed by machine learning algorithms.

5. **Instantiated and fitted the SVC model:** Created an instance of the Support Vector Classifier with a linear kernel. Adjusted the class imbalance in the data using the "balanced" parameter to give equal importance to both toxic and non-toxic comments during model training. Fitted the model on the train data.

6. **Performed hyperparameter tuning:** Optimized the model using GridSearchCV and StratifiedKFold for cross-validation. Provided a parameter grid for the 'C' parameter in the SVC model. The GridSearchCV function was searched for the best combination of hyperparameters using cross-validation, and the StratifiedKFold function helped address class imbalance issues.

7. **Evaluated the model:** Assessed the performance of the optimized SVC model using accuracy, recall, and f1_score metrics. These metrics provided insights into the model's ability to accurately classify comments as toxic or non-toxic and the balance between false positives and false negatives.

8. **Identified the most prominent terms in toxic comments:** Used the best estimator from the grid search to predict toxic comments in the test dataset. Extracted the terms from these toxic comments and created a list of the most frequent terms. This provided an understanding of the words and phrases that were most commonly associated with toxic comments on Wikipedia Talk edit pages.

**Results:**

Using the SVC model with hyperparameter tuning and a 70:30 dataset split, an accuracy of 86% was achieved in toxic text classification using 4000 features. This accuracy was further improved to over 95% when using all the features available.

**Analysis and Performance:**

The model demonstrated excellent performance with an accuracy of 86% on the test set and a recall rate of 0.63 for toxic comments. The f1_score was also satisfactory, which indicates a good balance between precision and recall.

**Optimization:**

Grid search was utilized to perform hyperparameter tuning, resulting in the best parameter being C=1000. This optimization led to an improvement in the model's recall and f1_score, which allowed for better detection of toxic comments.

**Conclusion:**
This project successfully developed a text classifier using NLP text processing and the Support Vector Classifier (SVC) model in Python to determine toxic comments in the Talk edit pages on Wikipedia. The model achieved an accuracy of 86% in toxic text classification using 4000 features, and over 95% when using all features. Hyperparameter tuning and addressing class imbalance significantly improved the model's recall and f1_score, leading to better detection of toxic comments. The model also identified the top terms from the toxic comments, providing valuable insights for further analysis and potential mitigation strategies.

**Citations:**

1. pandas development team. (n.d.). pandas: Powerful data structures for data analysis, time series, and statistics. Retrieved from https://pandas.pydata.org/

2. Bird, S., Klein, E., & Loper, E. (n.d.). Natural language toolkit. Retrieved from https://www.nltk.org/

3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Retrieved from https://scikit-learn.org/

## "Sentiment Analysis of Twitter Comments on the Ukraine War using TextBlob

**Introduction:**

Sentiment Analysis is a critical tool in the domain of Natural Language Processing (NLP) and has numerous applications in analyzing online opinions,
reviews, and social media. In this project, a Sentiment Analysis model is developed using TextBlob, a Python library for processing textual data. 

**Objective:** The primary objective is to analyze a dataset of tweets and categorize them into negative, neutral, and positive sentiment classes
based on the generated polarity score. The threshold settings for sentiment class division are as follows:

Negative: Polarity < 0
Neutral: Polarity == 0
Positive: Polarity > 0

**TextBlob Function:**

TextBlob is an open-source Python library that provides a simple API for natural languages processing tasks, such as part-of-speech tagging,
noun phrase extraction, sentiment analysis, classification, translation, and more. It is built on top of the NLTK (Natural Language Toolkit)
library and provides additional functionality and ease of use for handling textual data.

The TextBlob library offers several pre-trained functions, one of which is the sentiment analysis function. The sentiment analysis function 
in TextBlob works based on a pre-built lexicon that contains a list of words, each with their corresponding polarity and subjectivity scores.
Polarity scores range from -1 (negative sentiment) to +1 (positive sentiment), while subjectivity scores range from 0 (objective) to 1 (subjective).
The function calculates the sentiment of a given text by analyzing the words in the text and aggregating their scores.

Here's a brief overview of how TextBlob's sentiment analysis function works:

    a. Tokenization: The input text is split into individual words (tokens).
    b. Sentiment Calculation: Each token's polarity and subjectivity scores are looked up in the pre-built lexicon.
    If a word is not found in the lexicon, it's assigned a default score of zero.
    c. Aggregation: The scores for all tokens in the text are aggregated to compute the overall polarity and subjectivity
    scores for the entire text.
    
**Step-by-Step Process:**

  1. Imported the necessary libraries, such as pandas, numpy, re, TextBlob, matplotlib, and WordCloud.
  
  2. Loaded the dataset, a CSV file containing tweets using pandas and created a data frame.
  
  3. Performed data cleaning by removing unnecessary elements such as @mentions, hashtags, RT (retweets), and hyperlinks using regular expressions.
  
  4. Defined functions to calculate subjectivity and polarity scores for each tweet using TextBlob.
  
  5. Added the subjectivity and polarity scores to the Data Frame as new columns.
  
  6. Created a Word-Cloud visualization to display the most frequent words in the dataset.
  
  7. Defined a function to compute sentiment analysis based on the polarity score, and added a new column 'Analysis' to a data frame.
  
  8. Sorted and printed positive and negative tweets.
  
  9. Plotted the polarity and subjectivity scores using a scatterplot.
  
  10. Calculated and displayed the percentage of positive and negative tweets.
  
  11. Plotted the sentiment analysis results using a bar chart.
  
**Results and Analysis:**

The model categorizes the tweets into negative, neutral, and positive sentiment classes, displaying the following distribution:
Positive: 50%
Negative: 15%
Neutral: 35%

**Performance:**

TextBlob, a lexicon-based method, provides a simple yet effective approach to perform sentiment analysis on the dataset. Although it may not 
be as accurate as more complex machine learning-based methods, its ease of use and fast processing time make it suitable for initial exploratory
analysis.

**Optimization:**

In the future, the model can be improved by fine-tuning the lexicon, incorporating more complex algorithms, or even employing deep learning
techniques such as recurrent neural networks (RNNs) or transformers for more accurate sentiment analysis.

**Conclusion:**

In conclusion, this project successfully demonstrated the application of two distinct methods, a lexicon-based approach using TextBlob and
a deep learning model using BERT, for sentiment analysis on a dataset of tweets. The TextBlob method provided a straightforward and fast way
to categorize tweets into negative, neutral, and positive sentiments, while the BERT-based model offered a more advanced and accurate solution
for sentiment classification.

**Citations:**

1. Loria, S. (2018). TextBlob: Simplified Text Processing. Retrieved from https://textblob.readthedocs.io/en/dev/
  

import torch
from transformers import pipeline

#This is a very minor testing area for different LLMs. If any of you girls wish to try out different models, 
#feel free to do it using this test
sentiment_analysis = pipeline(task="sentiment-analysis",
                              model="nlptown/bert-base-multilingual-uncased-sentiment")

def sentiment_analyse(article):
    return sentiment_analysis(article)

    


if __name__ == "__main__":
    print(sentiment_analyse("I love you"))
    print(sentiment_analyse("Are we going to die?"))
    print(sentiment_analyse("That is disgusting"))
    print(sentiment_analyse("I hate you"))
    print(sentiment_analyse("The tanks are on the border now"))
# To run enter "streamlit run streamlit_nlp.py" on the terminal
# Core Pkgs
import streamlit as st
import os
import plotly.express as px
from textblob import TextBlob
from collections import Counter   
import pandas as pd
from language_tool_python import LanguageTool

# NLP Pkgs
from textblob import TextBlob
import spacy
from summarizer import Summarizer
import speech_recognition as sr


# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Function for Sumy Summarization
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary if "This section does not cite any sources." not in sentence]
    result = ' '.join(summary_list)
    return result

def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities

# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData

# Function for Extractive Summarization using BERT
def bert_summarizer(docx):
    model = Summarizer()
    summary = model(docx)
    return summary


def main():
    """ NLP based app with streamlit for text summarization, tokenization, sentiment analysis and voice input to text conversion along with summarization (Lekanakin's code used as reference)"""

    st.markdown("""
        #### Description
        + This is a Natural Language Processing(NLP) Based App useful for basic NLP task
        Tokenization , Lemmatization, Named Entity Recognition (NER), Sentiment Analysis, Text Summarization and Speech to text conversion- along with speech to text summarization. Click any of the checkboxes to get started.
        """)

    # Summarization
    if st.checkbox("Get the summary of your text"):
        st.subheader("Summarize Your Text")

        message = st.text_area("Enter Text", "Type Here....")
        if st.button("Summarize"):
            st.text("Using BERT Extractive Summarizer ..")
            summary_result = bert_summarizer(message)
            st.success(summary_result)

    # Sentiment Analysis Visualization
    if st.checkbox("Get the Sentiment Score of your text"):
        st.subheader("Identify Sentiment in your Text")

        message = st.text_area("Enter Text", "Type Here...")
        if st.button("Analyze"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)


    # Entity Extraction
    if st.checkbox("Get the Named Entities"):
        st.subheader("Identify Entities in your text")

        message = st.text_area("Enter Text", "Type Here..")
        if st.button("Analyze"):
            entity_results = entity_analyzer(message)
            st.json(entity_results)
            

    # Tokenization
    if st.checkbox("Get the Tokens and Lemma of text- along with word count + frequency"):
        st.subheader("Tokenize Your Text")

        message = st.text_area("Enter Text", "Type Here.")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)
        
        # Token Frequency Visualization
        tokens = [data.split(',')[0].split(':')[1].strip('"') for data in nlp_result]
        token_counter = Counter(tokens)
        token_df = pd.DataFrame(token_counter.items(), columns=['Token', 'Frequency'])
        
        st.subheader("Token Frequencies")
        st.dataframe(token_df)
        
        # Word Frequencies
        words = [token for token in tokens if token.isalpha()]
        word_counter = Counter(words)
        
        st.subheader("Word Frequencies")
        st.write("Total Words:", len(words))
        st.write("Unique Words:", len(word_counter))
        st.dataframe(pd.DataFrame(word_counter.items(), columns=['Word', 'Frequency']))


    # Voice Interaction
    if st.checkbox("Use Voice Input- to convert voice to text"):
        st.subheader("Speak Your Text")

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        with st.spinner("Listening..."):
            # Use microphone as the source
            with sr.Microphone() as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                st.text("Recognizing...")
                message = recognizer.recognize_google(audio)
                st.text_area("Voice Input", message)
            except sr.UnknownValueError:
                st.warning("Could not understand audio. Please try again. :(")
            except sr.RequestError as e:
                st.error(f"Error: {e}")


    # Speech to Text Summarization
    if st.checkbox("Speech input for text summarization"):
        st.subheader("Speech to Text Summarization")
        if st.button("Click to record and summarize"):
            with st.spinner("Listening..."):
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)

                try:
                    st.text("Recognizing...")
                    speech_text = recognizer.recognize_google(audio)
                    print("Transcribed Speech:", speech_text)  # Print the transcribed speech
                    speech_text_area = st.text_area("Transcribed Speech", speech_text)
                except sr.UnknownValueError:
                    st.warning("Could not understand audio. Please try again. :(")
                except sr.RequestError as e:
                    st.error(f"Error: {e}")

                if 'speech_text' in locals():
                    # Summarize using BERT
                    st.text("Using BERT Extractive Summarizer ..")
                    summary_result = bert_summarizer(speech_text)  
                    st.write("Summary:")
                    st.write(summary_result)
                else:
                    st.warning("Please transcribe speech before summarizing.")

                
    st.sidebar.text("NLP for everyone.")
    st.sidebar.info("Use this tool to get the Sentiment score, tokens, lemma, Named Entities, Summary of your text and Convert speech to text- along with speech to text summarization.")

if __name__ == '__main__':
    main()

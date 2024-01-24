import spacy
# import glob in case user enters a file pattern
import glob
# import shutil in case user enters a compressed archive (.zip, .tar, .tgz etc.); this is more general than zipfile
import shutil
# import plotly for making graphs
import plotly.express as px
# import wordcloud for making wordclouds
import wordcloud
# import json
import json 
# import re
import re
#import pyate
import pyate 
#import pipline
from transformers import pipeline
#import sentence transformers
from sentence_transformers import SentenceTransformer
# Prevent stochastic behavior
from umap import UMAP
# Set minimum cluster size
from hdbscan import HDBSCAN
# "Improve" default representation
from sklearn.feature_extraction.text import CountVectorizer
# Use multiple representations
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
# Make topic model using all of this setup
from bertopic import BERTopic
#importing Autotokenizer
from transformers import AutoTokenizer

class span:
    """A simple class that models a text span."""
    

    def __init__(self, document, start, end): 
        """
        Initialize any new instances of class span with the following attributes
        
        :param document: the text the span is part of
        :type document: str
        :param start: the start character of the span in the text
        :type start: int
        :param end: the end character of the span in the text
        :type end: int
        """
        self.document = document
        self.start = start
        self.end = end
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end


    def length(self): 
        """
        Calculates the length of the span
        
        :returns: the length of the span
        :rtype: int
        """

        return len(self)
    

    def text(self):
        """
        Returns the text of the span
        
        :returns: the text of the span
        :rtype: str
        """
        return self.document[self.start:self.end+1]

class counter(dict):
    def __init__(self, list_of_items, top_k=-1):
        """Makes a counter.

        :param list_of_items: the items to count
        :type list_of_items: list
        :param top_k: the number you want to keep
        :type top_k: int
        :returns: a counter
        :rtype: counter
        """
        super().__init__()
        for item in list_of_items:
            self.add_item(item)
        if top_k > 0:
            self.reduce_to_top_k(top_k)

     
        
    def add_item(self, item):
        """Adds an item to the counter.

        :param item: thing to add
        :type item: any
        """
      
        if item not in self:
            self[item] = 0
        self[item] += 1

        
    def get_counts(self):
        """Gets the counts from this counter.

        :returns: a list of (item, count) pairs
        :type item: list[tuple]
        """
  
        return list(sorted(self.items(), key=lambda x:x[1]))
    
    def reduce_to_top_k(self, top_k):
        """Gets the top k most frequent items.

        :param top_k: the number you want to keep
        :type top_k: int
        """
        top_k = min([top_k, len(self)])
        sorted_keys = sorted(self, key=lambda x: self[x])
        for i in range(0, (len(self)-top_k)+1):
            self.pop(sorted_keys[i])

class corpus(dict):
    nlp = spacy.load('en_core_web_md')          
    nlp.add_pipe("combo_basic")
    classifier = pipeline("sentiment-analysis")
    summarizer_model = pipeline("summarization", model = "Falconsai/text_summarization", min_length=5, max_length = 30)
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    keybert_model = KeyBERTInspired()
    pos_model = PartOfSpeech("en_core_web_md")
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    sentence_sentiment = pipeline("text-classification", model="sst-model", tokenizer=tokenizer)
    

    def get_sentence_level_sentiment(self, doc_id):
        """Gets sentence level sentiment from a document in the corpus

        :param doc_id: a document_id from the corpus
        :returns: a list of sentence label pairs
        :rtype: list
        """

        sentiment_dict = dict()
        for sentence in self[doc_id]['doc'].sents:
            sentiment_label = corpus.sentence_sentiment(sentence.text)[0]['label']
            sentiment_dict[sentence] = sentiment_label
        return list(sentiment_dict.items())
    
    def render_document_sentiments(self, doc_id):
        """Render a document's token and entity counts as a table. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str
        """
        sent_table = '| Sentence | Sentiment | Aspect | \n | ------------ | -------- | -------- |'
       
        sentence_list = self.get_sentence_level_sentiment(doc_id)
        for sentence, sent_label in sentence_list:
            key_phrase_list = self.get_keyphrase_doc(str(sentence))
            sent_table += "\n| " + str(sentence) + " | " + sent_label + " | " + key_phrase_list[0] + " |"
        sent_table += '\n| Document Label | Document Sentiment |\n | ------------ | ----- |'
        sent_table += "\n| " + self[doc_id]['sentiment analysis'][0]['label'] + " | " + str(self[doc_id]['sentiment analysis'][0]['score']) + " |"
        return sent_table        
                    
    def __init__(self, name=''):
        """Creates or extends a corpus.

        :param name: the name of this corpus
        :type name: str
        :returns: a corpus
        :rtype: corpus
        """
        super().__init__()
        self.name = name 
       
    def get_documents(self):
        """Gets the documents from the corpus.

        :returns: a list of spaCy documents
        :rtype: list
        """
 
        return [item['doc'] for item in self.values()]
   
    def get_document(self, id):
        """Gets a document from the corpus.

        :param id: the document id to get
        :type id: str
        :returns: a spaCy document
        :rtype: (spaCy) doc
        """

        return self[id]['doc'] if id in self and 'doc' in self[id] else None 
                         
    def get_metadatas(self):
        """Gets the metadata for each document from the corpus.

        :returns: a list of metadata dictionaries
        :rtype: list[dict]
        """
  
        return [item['metadata'] for item in self.values()]

    def get_metadata(self, id):
        """Gets a metadata from the corpus.

        :param id: the document id to get
        :type id: str
        :returns: a metadata dictionary
        :rtype: dict
        """
   
        return self[id]['metadata'] if id in self and 'metadata' in self[id] else None
    
    def get_document_texts(self):
        pair_dict = dict()
        for i in self.keys():
            pair_dict[i] = self.get_document(i)
        return list(pair_dict.items())
    
    
    def update_document_metadata(self, id, metadata_value_pair):
        """
        Updates the metadata corpus.

        :param id: the document id
        :type id: str
        :param doc: the document itself
        :type doc: (spaCy) doc
        :param metadata: the document metadata
        :type metadata: dict
        """
        try:
            for i in metadata_value_pair.keys():
                self[id]['metadata'][i] = metadata_value_pair[i]
        except:
            print("ID not in corpus") 
                         
    def add_document(self, id, doc, metadata={}):
        """Adds a document to the corpus.

        :param id: the document id
        :type id: str
        :param doc: the document itself
        :type doc: (spaCy) doc
        :param metadata: the document metadata
        :type metadata: dict
        """
        self[id] = {'doc': self.nlp(doc), 'metadata': metadata, 'sentiment analysis': corpus.classifier(str(doc)), 'summarizer model': corpus.summarizer_model(str(doc))}
        
    def get_token_counts(self, tags_to_exclude = ['PUNCT', 'SPACE'], top_k=-1):
        """Builds a token frequency table.

        :param tags_to_exclude: (Coarse-grained) part of speech tags to exclude from the results
        :type tags_to_exclude: list[string]
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
   
        token_list = []
        for doc_id in self.get_documents():
            token_list.extend([token.text for token in doc_id if token.pos_ not in tags_to_exclude])
        return counter(token_list, top_k).get_counts()
       

    def get_entity_counts(self, tags_to_exclude = ['QUANTITY'], top_k=-1):
        """Builds an entity frequency table.

        :param tags_to_exclude: named entity labels to exclude from the results
        :type tags_to_exclude: list[string]
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """

        entity_list = []
        for doc_id in self.get_documents():
            entity_list.extend([ent.text for ent in doc_id.ents if ent.label_ not in tags_to_exclude])
        return counter(entity_list, top_k).get_counts()

    def get_noun_chunk_counts(self, top_k=-1):
        """Builds a noun chunk frequency table.

        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        chunk_list = []
        for doc_id in self.get_documents():
            chunk_list.extend([chunk.text for chunk in doc_id.noun_chunks])
        return counter(chunk_list, top_k).get_counts()
    
    def get_paragraph_counts(self, top_k=-1):
        paragraphs = []
        start = 0 

        for doc in self.get_documents():
            spacy_paragraphs = []
            for token in doc: 
                if token.is_space and token.text.count("\n") > 1:
                    spacy_paragraphs.append(span(doc, start, token.i))
                    start = token.i + 1
            paragraphs.extend(spacy_paragraphs)
        return counter(paragraphs).get_counts()

    def get_metadata_counts(self, key, top_k=-1):
        """Gets frequency data for the values of a particular metadata key.

        :param key: a key in the metadata dictionary
        :type key: str
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
 
        metadata_list = []
        metadata_list.extend([metadata[key] for metadata in self.get_metadatas() if key in metadata])
        return counter(metadata_list, top_k).get_counts()

    def get_sentiment_counts(self, top_k=-1):
        """
        Gets sentiment frequencies.

        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """

        sentiment_list =[]
        for id in self.keys():
            sentiment_list.extend([sent['label'] for sent in self[id]['sentiment analysis']])
        return counter(sentiment_list).get_counts()
    
    def get_keyphrases_counts(self, top_k = 25):
        """
        Builds a noun chunk frequency table.

        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        key_phrase_list = []
        for doc_id in self.get_documents():
            key_phrase_list.extend(list(doc_id._.combo_basic.keys()))
        return counter(key_phrase_list, top_k = top_k).get_counts()

    def get_keyphrase_doc(self, sent):
        """
        Get a keyphrases from a singular doc.

        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        doc = self.nlp(sent)
        return list(doc._.combo_basic.keys())
    

    def get_token_statistics(self):
        """Prints summary statistics for tokens in the corpus, including: number of documents; number of sentences; number of tokens; number of unique tokens.
        
        :returns: the statistics report
        :rtype: str
        """
        
        text = f'Documents: %i\n' % len(self)
        text += f'Sentences: %i\n' % sum([len(list(doc.sents)) for doc in self.get_documents()])
        token_counts = self.get_token_counts()
        text += f'Tokens: %i\n' % sum([x[1] for x in token_counts])
        text += f"Unique tokens: %i\n" % len(token_counts)
        return text

    def get_entity_statistics(self):
        """Prints summary statistics for entities in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """

        text = f'Documents: %i\n' % len(self)
        text += f'Sentences: %i\n' % sum([len(list(doc.sents)) for doc in self.get_documents()])
        entity_counts = self.get_entity_counts()
        text += f'Entities: %i\n' % sum([x[1] for x in entity_counts])
        text += f"Unique Entities: %i\n" % len(entity_counts)
        return text
        
    def get_noun_chunk_statistics(self):
        """Prints summary statistics for noun chunks in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """

        text = f'Documents: %i\n' % len(self)
        text += f'Sentences: %i\n' % sum([len(list(doc.sents)) for doc in self.get_documents()])
        noun_counts = self.get_noun_chunk_counts()
        text += f'Noun Chunks: %i\n' % sum([x[1] for x in noun_counts])
        text += f"Unique Noun Chunks: %i\n" % len(noun_counts)
        return text
    
    def get_key_phrase_statistics(self):
        """Prints summary statistics for keyphrases in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """

        text = f'Documents: %i\n' % len(self)
        text += f'Sentences: %i\n' % sum([len(list(doc.sents)) for doc in self.get_documents()])
        key_phrase_counts = self.get_keyphrases_counts()
        text += f'Key Phrases: %i\n' % sum([x[1] for x in key_phrase_counts])
        text += f"Unique Key Phrases: %i\n" % len(key_phrase_counts)
        return text
    
    def get_sentiment_statistics(self):
        """Prints summary statistics for sentiments in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """
  
        text = f'Documents: %i\n' % len(self)
        text += f'Sentences: %i\n' % sum([len(list(doc.sents)) for doc in self.get_documents()])
        """
        pos_count = 0
        neg_count = 0 
        neutral_count = 0      
        for id in self.keys():
            for sent in self[id]['sentiment analysis']:
                if sent['label'] == 'POSITIVE':
                    pos_count += 1
                elif sent['label'] == 'NEGATIVE':
                    neg_count += 1 
                else:
                    neutral_count += 1  
        """
        sentiment_counts = self.get_sentiment_counts()
        text += f'Positive Documents: %i\n' % sum([x[1] for x in sentiment_counts if x[0] == 'POSITIVE' ])
        text += f"Neutral Documents: %i\n" % sum([x[1] for x in sentiment_counts if x[0] == 'NEUTRAL' ])
        text += f"Negative Documents: %i\n" % sum([x[1] for x in sentiment_counts if x[0] == 'NEGATIVE' ])
        return text       
    
    def get_basic_statistics(self):
        """Prints summary statistics for the corpus.
        
        :returns: the statistics report
        :rtype: str
        """
        para_counts = self.get_paragraph_counts()
        return f' Statistics: %s\n ' % self.get_token_statistics() + self.get_entity_statistics() + self.get_noun_chunk_statistics() + self.get_sentiment_statistics()  + "Paragraphs: " + str(len(para_counts))

    def plot_counts(self, counts, file_name):
        """Makes a bar chart for counts.

        :param counts: a list of item, count tuples
        :type counts: list
        :param file_name: where to save the plot
        :type file_name: string
        """
        fig = px.bar(x=[x[0] for x in counts], y=[x[1] for x in counts])
        fig.write_image(file_name)

    def plot_token_frequencies(self, tags_to_exclude=['PUNCT', 'SPACE'], top_k=25):
        """Makes a bar chart for the top k most frequent tokens in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        """
    
        reduced_counts = counter(self.get_token_counts(tags_to_exclude, top_k))
        self.plot_counts(reduced_counts, 'token_counts.png')

    def plot_entity_frequencies(self, tags_to_exclude=['QUANTITY'], top_k=25):
        """Makes a bar chart for the top k most frequent entities in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
       """
    
        reduced_entity_counts = counter(self.get_entity_counts(tags_to_exclude, top_k))
        self.plot_counts(reduced_entity_counts, 'entity_counts.png')
     
    def plot_noun_chunk_frequencies(self, top_k=25):
        """Makes a bar chart for the top k most frequent noun chunks in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        """
 
        reduced_chunk_counts = counter(self.get_noun_chunk_counts(top_k))
        self.plot_counts(reduced_chunk_counts, 'noun_chunk_counts.png')
     
    def plot_metadata_frequencies(self, key, top_k=25):
        """Makes a bar chart for the frequencies of values of a metadata key in a corpus.

        :param key: a metadata key
        :type key: str        
        :param top_k: the number to keep
        :type top_k: int
        """
  
        reduced_meta_counts = counter(self.get_metadata_counts(key, top_k))
        self.plot_counts(reduced_meta_counts, key + '.png')
 
    def plot_word_cloud(self, counts, name):
        """Plots a word cloud.

        :param counts: a list of item, count tuples
        :type counts: list
        :param file_name: where to save the plot
        :type file_name: string
        :returns: the word cloud
        :rtype: wordcloud
        """
        wc = wordcloud.WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(dict(counts))
        cloud = px.imshow(wc)
        cloud.update_xaxes(showticklabels=False)
        cloud.update_yaxes(showticklabels=False)
        return cloud

    def plot_token_cloud(self, tags_to_exclude=['PUNCT', 'SPACE']):
        """Makes a word cloud for the frequencies of tokens in a corpus.

        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        :returns: the word cloud
        :rtype: wordcloud
        """
      
        token_counts = self.get_token_counts(tags_to_exclude)
        return self.plot_word_cloud(token_counts, "token_cloud.png")
    
    def plot_sentiment_cloud(self, tags_to_exclude=['PUNCT', 'SPACE']):
        """Makes a word cloud for the frequencies of tokens in a corpus.

        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        :returns: the word cloud
        :rtype: wordcloud
        """
      
        sentiment_counts = self.get_sentiment_counts()
        return self.plot_word_cloud(sentiment_counts, "sentiment_cloud.png")
 
    def plot_entity_cloud(self, tags_to_exclude=['QUANTITY']):
        """Makes a word cloud for the frequencies of entities in a corpus.
 
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        :returns: the word cloud
        :rtype: wordcloud
        """

        entity_counts = self.get_entity_counts(tags_to_exclude)
        return self.plot_word_cloud(entity_counts, "entity_cloud.png")

    def plot_noun_chunk_cloud(self):
        """Makes a word cloud for the frequencies of noun chunks in a corpus.

        :returns: the word cloud
        :rtype: wordcloud
        """

        chunk_counts = self.get_noun_chunk_counts()
        return self.plot_word_cloud(chunk_counts, "chunk_cloud.png")

        
    def render_doc_markdown(self, doc_id):
        """Render a document as markdown. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str

        """
        doc = self.get_document(doc_id)
        text = '# ' + doc_id + '\n\n'
        for token in doc:
            if token.pos_ == 'NOUN':
                text = text + '**' + token.text + '**'
            elif token.pos_ == 'VERB':
                text = text + '*' + token.text + '*'
            else:
                text = text + token.text
            text = text + token.whitespace_
        return text

    def render_doc_table(self, doc_id):
        """Render a document's token and entity annotations as a table. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str
        """
        doc = self.get_document(doc_id)
        tokens_table = "| Tokens | Lemmas | Coarse | Fine | Shapes | Morphology |\n| ------ | ------ | ------ | ---- | ------ | ---------- | \n"
        for token in doc:
            if token.pos_ != 'SPACE':
                tokens_table  = tokens_table + "| " + token.text + " | " + token.lemma_ + " | " + token.pos_ + " | " + token.tag_ + " | " + token.shape_ + " | " + re.sub(r'\|', '#', str(token.morph)) + " |\n"
        entities_table = "| Text | Type |\n| ---- | ---- |\n"
        for entity in doc.ents:
            entities_table = entities_table + "| " + entity.text + " | " + entity.label_ + " |\n"
        return '## Tokens\n' + tokens_table + '\n## Entities\n' + entities_table

    def render_doc_statistics(self, doc_id):
        """Render a document's token and entity counts as a table. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str
        """
        doc = self.get_document(doc_id)
        stats = {}
        for token in doc:
            if token.pos_ != 'SPACE':
                if token.text + token.pos_ not in stats:
                    stats[token.text + token.pos_] = 0
                stats[token.text + token.pos_] = stats[token.text + token.pos_] + 1
        for entity in doc.ents:
            if entity.text + entity.label_ not in stats:
                stats[entity.text + entity.label_] = 0
            stats[entity.text + entity.label_] = stats[entity.text + entity.label_] + 1
        text = '| Token/Entity | Count |\n | ------------ | ----- |\n'
        for key in sorted(stats.keys()):
            text += '| ' + key + ' | ' + str(stats[key]) + ' |\n'
        return text

    @classmethod
    def load_textfile(cls, file_name, my_corpus=None):
        """Loads a textfile into a corpus.

        :param file_name: the path to a text file
        :type file_name: string
        :param my_corpus: a corpus
        :type my_corpus: corpus
        :returns: a corpus
        :rtype: corpus
         """
        if my_corpus == None:
            my_corpus = corpus()
        with open(file_name, encoding= 'utf-8') as f:
            my_corpus.add_document(file_name, cls.nlp(' '.join(f.readlines())))
        return my_corpus

    def topic_model(self, graph_key):
        full_texts = [self[x]['doc'].text for x in self]*50
        topic_model = self.build_topic_model()
        embeddings = self.embedding_model.encode(full_texts, show_progress_bar=True)
        topic_model.fit_transform(full_texts, embeddings)
        if graph_key == 'topic':
            return topic_model.visualize_topics()
        elif graph_key =='document':
            reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
            return topic_model.visualize_documents(full_texts, reduced_embeddings=reduced_embeddings)
    


    @classmethod
    def build_topic_model(cls):
    
        representation_model = {
        "KeyBERT": cls.keybert_model,
        "MMR": cls.mmr_model,
        "POS": cls.pos_model}
        
        topic_model = BERTopic(
        # Pipeline models
        embedding_model= cls.embedding_model,
        umap_model= cls.umap_model,
        hdbscan_model= cls.hdbscan_model,
        vectorizer_model = cls.vectorizer_model,
        representation_model = representation_model,

        # Hyperparameters
        top_n_words=10,
        verbose=True,
        nr_topics="auto"
        )

        return topic_model
        

    @classmethod  
    def load_jsonl(cls, file_name, my_corpus=None):
        """Loads a jsonl file into a corpus.

        :param file_name: the path to a jsonl file
        :type file_name: string
        :param my_corpus: a my_corpus
        :type my_corpus: my_corpus
        :returns: a my_corpus
        :rtype: my_corpus
         """
        if my_corpus == None:
            my_corpus = corpus()
        with open(file_name, encoding='utf-8') as f:
            for line in f.readlines():
                js = json.loads(line)
                if 'fullText' in js and 'id' in js:
                    my_corpus.add_document(js["id"], cls.nlp(''.join(js["fullText"])), metadata=js)
        return my_corpus
    

    @classmethod   
    def load_compressed(cls, file_name, my_corpus=None):
        """Loads a zipfile into a corpus.

        :param file_name: the path to a zipfile
        :type file_name: string
        :param my_corpus: a corpus
        :type my_corpus: corpus
        :returns: a corpus
        :rtype: corpus
       """
        if my_corpus == None:
            my_corpus = corpus()
        shutil.unpack_archive(file_name, 'temp')
        for file_name2 in glob.glob('temp/*'):
            cls.build_corpus(file_name2, my_corpus = my_corpus)
        shutil.rmtree("temp")
        return my_corpus

    @classmethod
    def build_corpus(cls, pattern, my_corpus=None):
        """Builds a corpus from a pattern that matches one or more compressed or text files.

        :param pattern: the pattern to match to find files to add to the corpus
        :type file_name: string
        :param my_corpus: a corpus
        :type my_corpus: corpus
        :returns: a corpus
        :rtype: corpus
        if my_corpus == None:
            my_corpus = corpus(pattern)
        try:
            for file_name in glob.glob(pattern):
                if file_name.endswith('.zip') or file_name.endswith('.tar') or file_name.endswith('.tgz'):
                    cls.load_compressed(file_name, my_corpus)
                elif file_name.endswith('.jsonl'):
                    cls.load_jsonl(file_name, my_corpus)
                else:
                    cls.load_textfile(file_name, my_corpus)
        except Exception as e:
            print(f"Couldn't load % s due to error %s" % (pattern, str(e)))
        return my_corpus

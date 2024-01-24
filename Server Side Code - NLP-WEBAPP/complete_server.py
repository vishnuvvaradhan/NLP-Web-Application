# import corpus from spacy_on_corpus
from spacy_on_corpus import corpus

# import anvil server
import anvil.server

# make a corpus instance called my_corpus
my_corpus = corpus()


# make a corpus instance called my_corpus

@anvil.server.callable
def load_file(filename, file_contents):
    """Call build_corpus on file_contents, giving it name filename
    
    :param filename: the filename we want to store file_contents in
    :type filename: str
    :param file_contents: the contents we want to use to build / augment my_corpus
    :type file_contents: byte stream
    """
    # first we write file_contents to a file which will have name inputs/filename
    with open('inputs/' + filename, 'wb') as f:
        f.write(file_contents.get_bytes())
    # You call build_corpus on inputs/filename, giving it my_corpus as a keyword argument
    #file_name = os.path.basename('inputs/' + file_name)
    corpus.build_corpus('inputs/' + filename, my_corpus = my_corpus)

@anvil.server.callable
def add_document(text):
    """Add a document to my_corpus using contents.
    
    :param text: the text we want to add to my_corpus
    :type text: str
    """
    # You add a document to my_corpus using text and give it a unique id
    my_corpus.add_document(str(len(my_corpus)), text)
   

@anvil.server.callable
def clear():
    """Empty my_corpus."""
    # You implement this using an instance method of dict
    my_corpus.clear()


@anvil.server.callable
def get_corpus_tokens_counts(top_k=25):
    """Get the token counts from my_corpus.
    
    :param top_k: the top_k tokens to return
    :type top_k: int
    :returns: a list of pairs (item, frequency)
    :rtype: list
    """
    # You return the token counts
    return my_corpus.get_token_counts(tags_to_exclude= [] , top_k=top_k)

@anvil.server.callable
def get_corpus_entities_counts(top_k=25):
    """Get the entity counts from my_corpus.
    
    :param top_k: the top_k entities to return
    :type top_k: int
    :returns: a list of pairs (item, frequency)
    :rtype: list
    """
    # You return the entity counts
    return my_corpus.get_entity_counts(tags_to_exclude=[], top_k=top_k)

@anvil.server.callable
def get_corpus_noun_chunks_counts(top_k=25):
    """Get the noun chunk counts from my_corpus.
    
    :param top_k: the top_k noun chunks to return
    :type top_k: int
    :returns: a list of pairs (item, frequency)
    :rtype: list
    """
    # You return the noun chunk counts
    return my_corpus.get_noun_chunk_counts(top_k=top_k)

@anvil.server.callable
def get_corpus_sentiment(top_k=25):
    """
    Get the sentiment counts from my_corpus.
    :param top_k: the top_k noun chunks to return
    :type top_k: int
    :returns: a list of pairs (item, frequency)
    :rtype: list
    """
    return my_corpus.get_sentiment_counts(top_k)

@anvil.server.callable
def get_corpus_tokens_statistics():
    """Get the token statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """
    # You return the token statistics
    return my_corpus.get_token_statistics()

@anvil.server.callable
def get_corpus_entities_statistics():
    """Get the entity statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """
    # You return the entity statistics
    return my_corpus.get_entity_statistics()

@anvil.server.callable
def get_corpus_sentiment_statistics():
    """Get the entity statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """
    # You return the entity statistics
    return my_corpus.get_sentiment_statistics()


@anvil.server.callable
def get_corpus_noun_chunks_statistics():
    """Get the noun chunk statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """
    # You return the noun chunk statistics
    return my_corpus.get_noun_chunk_statistics()

@anvil.server.callable
def get_corpus_statistics():
    """
    Get the general statistics from my_corpus includes sentiments of each document.
    :returns: basic statistics suitable for printing
    :rtype: str
    """
    return my_corpus.get_basic_statistics()

@anvil.server.callable
def get_corpus_keyphrase_statistics():
    """Get the keyphrase statistics from my_corpus.
    :returns: basic statistics suitable for printing
    :rtype: str
    """
    # You return the noun chunk statistics
    return my_corpus.get_key_phrase_statistics()

@anvil.server.callable
def get_token_cloud():
    """Get the token cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """
    # You get the token counts
    token_counts = get_corpus_tokens_counts()
    # You make the word cloud if token_counts is not None
    if token_counts != None:
        return my_corpus.plot_token_cloud()

@anvil.server.callable
def get_entity_cloud():
    """Get the entity cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """
    # You get the entity counts
    entity_counts = get_corpus_entities_counts()
    #entity_counts = my_corpus.get_entity_counts()
    # You make the entity cloud if entity_counts is not None
    if entity_counts != None:
        return my_corpus.plot_entity_cloud()

@anvil.server.callable
def get_noun_chunk_cloud():
    """Get the noun chunk cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """
    # You get the noun chunk counts
    noun_chunk_counts = get_corpus_noun_chunks_counts()

    # You make the noun chunk cloud if chunk_counts is not None
    if noun_chunk_counts != None:
        return my_corpus.plot_noun_chunk_cloud()

@anvil.server.callable
def get_sentiment_cloud():
    """Get the sentiment cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """
    # You get the noun chunk counts
    sentiment_counts = my_corpus.get_sentiment_counts()

    # You make the noun chunk cloud if chunk_counts is not None
    if sentiment_counts != None:
        return my_corpus.plot_sentiment_cloud()

@anvil.server.callable
def get_document_summarys(doc_id):
    """
    Get a document summary for a specific document
    
    :returns: a document summary string
    :rtype: str
    """
    if doc_id in my_corpus.keys():
        summary_dict = my_corpus[doc_id]['summarizer model'][0]
        return summary_dict['summary_text']
    
@anvil.server.callable
def get_topic_model_topics_plot():
    """
    Get a topic model plot for the corpus
    
    :returns: a topic model plot 
    :rtype: plot 
    """
    return my_corpus.topic_model('topic')

@anvil.server.callable
def get_topic_model_documents_plot():
    """
    Get a topic document model plot for the corpus
    
    :returns: the topic model document plot
    :rtype: plot 
    """
    return my_corpus.topic_model('document')



@anvil.server.callable
def clear():
    """
    when clear is clicked in the web-app it clears the statistics off the page

    :returns: an empty string
    :rtype: str 
    """
    return '                 '

@anvil.server.callable
def get_document_ids():
    """Get the ids of all document ids in the corpus.
    
    :returns: the document ids
    :rtype: list[str]
    """
    # You get the list of document ids in the corpus
    return list(my_corpus.keys())


@anvil.server.callable
def get_doc_markdown(doc_id):
    """Get the document markdown for a document in my_corpus.
    
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
    # You do it!
    return my_corpus.render_doc_markdown(doc_id)

@anvil.server.callable
def get_doc_table(doc_id):
    """Get the document table for a document in my_corpus.
    
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
    # You do it!
    return my_corpus.render_doc_table(doc_id)

@anvil.server.callable
def get_doc_statistics(doc_id):
    """Get the document statistics for a document in my_corpus.
    
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
    # You do it!
    return my_corpus.render_doc_statistics(doc_id)


@anvil.server.callable
def get_doc_sentiment_markdown(id):
    """Get the document sentiments for a document in my_corpus.
    
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
    return my_corpus.render_document_sentiments(id)
    


# YOUR ANVIL CALLABLES HERE

def run():
    """Run the server!"""  
    # connect
    anvil.server.connect("server_XCJN5ACUIZENMHX47X4SAWAF-E33GC75AVOG6TPZ2")
    # wait forever
    anvil.server.wait_forever()

# this says, if executing this on the command line like python server.py, run run()    
if __name__ == "__main__":
    run()

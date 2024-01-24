from spacy_on_corpus import corpus
import anvil.server


my_corpus = corpus()


@anvil.server.callable
def load_file(filename, file_contents):
    """
    :param filename: the filename we want to store file_contents in
    :type filename: str
    :param file_contents: the contents we want to use to build / augment my_corpus
    :type file_contents: byte stream
    """
    with open('inputs/' + filename, 'wb') as f:
        f.write(file_contents.get_bytes())
    corpus.build_corpus('inputs/' + filename, my_corpus = my_corpus)

@anvil.server.callable
def add_document(text):
    """
    :param text: the text we want to add to my_corpus
    :type text: str
    """
    my_corpus.add_document(str(len(my_corpus)), text)
   

@anvil.server.callable
def clear():
    """Empty my_corpus."""
    my_corpus.clear()


@anvil.server.callable
def get_corpus_tokens_counts(top_k=25):
    """
    :param top_k: the top_k tokens to return
    :type top_k: int
    :returns: a list of pairs (item, frequency)
    :rtype: list
    """
    return my_corpus.get_token_counts(tags_to_exclude= [] , top_k=top_k)

@anvil.server.callable
def get_corpus_entities_counts(top_k=25):
    """
    :param top_k: the top_k entities to return
    :type top_k: int
    :returns: a list of pairs (item, frequency)
    :rtype: list
    """

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
    
    return my_corpus.get_token_statistics()

@anvil.server.callable
def get_corpus_entities_statistics():
    """Get the entity statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """

    return my_corpus.get_entity_statistics()

@anvil.server.callable
def get_corpus_sentiment_statistics():
    """Get the entity statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """

    return my_corpus.get_sentiment_statistics()


@anvil.server.callable
def get_corpus_noun_chunks_statistics():
    """Get the noun chunk statistics from my_corpus.
    
    :returns: basic statistics suitable for printing
    :rtype: str
    """

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

    return my_corpus.get_key_phrase_statistics()

@anvil.server.callable
def get_token_cloud():
    """Get the token cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """

    token_counts = get_corpus_tokens_counts()
   
    if token_counts != None:
        return my_corpus.plot_token_cloud()

@anvil.server.callable
def get_entity_cloud():
    """Get the entity cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """

    entity_counts = get_corpus_entities_counts()
    #entity_counts = my_corpus.get_entity_counts()
   
    if entity_counts != None:
        return my_corpus.plot_entity_cloud()

@anvil.server.callable
def get_noun_chunk_cloud():
    """Get the noun chunk cloud for my_corpus.
    :returns: an image
    :rtype: plot
    """
    noun_chunk_counts = get_corpus_noun_chunks_counts()
    
    if noun_chunk_counts != None:
        return my_corpus.plot_noun_chunk_cloud()

@anvil.server.callable
def get_sentiment_cloud():
    """Get the sentiment cloud for my_corpus.
    
    :returns: an image
    :rtype: plot
    """
    # get the noun chunk counts
    sentiment_counts = my_corpus.get_sentiment_counts()
   
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
    :returns: an empty string
    :rtype: str 
    """
    return '                 '

@anvil.server.callable
def get_document_ids():
    """
    :returns: the document ids
    :rtype: list[str]
    """
    return list(my_corpus.keys())


@anvil.server.callable
def get_doc_markdown(doc_id):
    """
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
 
    return my_corpus.render_doc_markdown(doc_id)

@anvil.server.callable
def get_doc_table(doc_id):
    """
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """

    return my_corpus.render_doc_table(doc_id)

@anvil.server.callable
def get_doc_statistics(doc_id):
    """
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
   
    return my_corpus.render_doc_statistics(doc_id)


@anvil.server.callable
def get_doc_sentiment_markdown(id):
    """
    :param doc_id: a document id
    :type doc_id: str
    :returns: markdown
    :rtype: str
    """
    return my_corpus.render_document_sentiments(id)
    


def run():
    anvil.server.connect("server_XCJN5ACUIZENMHX47X4SAWAF-E33GC75AVOG6TPZ2")
    anvil.server.wait_forever()

if __name__ == "__main__":
    run()

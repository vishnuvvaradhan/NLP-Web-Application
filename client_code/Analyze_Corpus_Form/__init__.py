from ._anvil_designer import Analyze_Corpus_FormTemplate
from anvil import *
import anvil.server
import plotly.graph_objects as go

class Analyze_Corpus_Form(Analyze_Corpus_FormTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

  def corpus_statistics_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    print('in statistics!')
    print(self.corpus_attributes_chooser.selected_value)
    if self.corpus_attributes_chooser.selected_value == 'Tokens':
      self.corpus_markdown.content = anvil.server.call('get_corpus_tokens_statistics')
    elif self.corpus_attributes_chooser.selected_value == 'Entities':
      self.corpus_markdown.content = anvil.server.call('get_corpus_entities_statistics')
    elif self.corpus_attributes_chooser.selected_value == 'Chunks':
      self.corpus_markdown.content = anvil.server.call('get_corpus_noun_chunks_statistics')
    self.corpus_statistics_button.selected = False

  def plot_counts(self, counts, name):
    if counts:
      self.corpus_plot.data = [go.Bar(x = [x[0] for x in counts], y = [x[1] for x in counts], name = name)]
    else:
      self.corpus_plot.data = []
    self.corpus_plot.layout = {'title': name}

  def corpus_counts_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    counts = None
    if self.corpus_attributes_chooser.selected_value == 'Tokens':
      counts = anvil.server.call('get_corpus_tokens_counts', top_k=25)
      self.plot_counts(counts, 'Token Counts')
    elif self.corpus_attributes_chooser.selected_value == 'Entities':
      counts = anvil.server.call('get_corpus_entities_counts', top_k=25)
      self.plot_counts(counts, 'Entity Counts')
    elif self.corpus_attributes_chooser.selected_value == 'Chunks':
      counts = anvil.server.call('get_corpus_noun_chunks_counts', top_k=25)
      self.plot_counts(counts, 'Noun Chunk Counts')
    self.corpus_counts_button.selected = False


  def plot_cloud(self, cloud, name):
    if cloud:
      self.corpus_plot.figure = cloud 
      self.corpus_plot.layout = {'title': name, 'xaxis': {'showticklabels': False}, 'yaxis': {'showticklabels': False}}
    else:
      self.corpus_plot.figure = None
      
  def corpus_cloud_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    if self.corpus_attributes_chooser.selected_value == 'Tokens':
      wc = anvil.server.call('get_token_cloud')
      self.plot_cloud(wc, 'Tokens')
    elif self.corpus_attributes_chooser.selected_value == 'Entities':
      wc = anvil.server.call('get_entity_cloud')
      self.plot_cloud(wc, 'Tokens')
    elif self.corpus_attributes_chooser.selected_value == 'Chunks':
      wc = anvil.server.call('get_noun_chunk_cloud')
      self.plot_cloud(wc, 'Tokens')
    self.corpus_cloud_button.selected = False

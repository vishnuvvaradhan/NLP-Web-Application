from ._anvil_designer import Analyze_Topics_FormTemplate
from anvil import *
import anvil.server
import plotly.graph_objects as go

class Analyze_Topics_Form(Analyze_Topics_FormTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

  def plot_cloud(self, cloud, name):
    if cloud:
      self.corpus_plot.figure = cloud
      self.corpus_plot.layout = {'title': name, 'xaxis': {'showticklabels': False}, 'yaxis': {'showticklabels': False}}
    else:
      self.corpus_plot.figure = None
      
  def plot_cloud_2(self, cloud, name):
    if cloud:
      self.corpus_plot_2.figure = cloud
      self.corpus_plot_2.layout = {'title': name, 'xaxis': {'showticklabels': False}, 'yaxis': {'showticklabels': False}}
    else:
      self.corpus_plot_2.figure = None
      
  def corpus_topic_model_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    self.corpus_plot_2= False
    wc = anvil.server.call('get_topic_model_topics_plot')
    self.plot_cloud(wc, 'Topic')
    self.corpus_topic_model_button.selected = False

  def corpus_topic_documents_model_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    self.corpus_plot_2 = False
    wc = anvil.server.call('get_topic_model_documents_plot')
    self.plot_cloud(wc, 'Document & Topics')
    self.orpus_topic_documents_model_button.selected = False

  def corpus_display_both_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    #self.corpus_plot.content.clf()
    #self.corpus_plot_2.content.clf()
   
    wc = anvil.server.call('get_topic_model_topics_plot')
    self.plot_cloud(wc, 'Topic')
    
    wc2 = anvil.server.call('get_topic_model_documents_plot')
    self.plot_cloud_2(wc2, 'Document & Topics')
    self.corpus_display_both_button.selected = False
    



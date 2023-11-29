from ._anvil_designer import Analyze_Document_FormTemplate
from anvil import *
import anvil.server

class Analyze_Document_Form(Analyze_Document_FormTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)
    # Any code you write here will run before the form opens.
    self.document_chooser.items = anvil.server.call('get_document_ids')

  def document_summary_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    if self.document_chooser.selected_value != None:
      self.document_markdown.content = anvil.server.call('get_document_summary', self.document_chooser.selected_value)
    self.document_render_button.selected = False
 

  def document_render_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    if self.document_chooser.selected_value != None:
      self.document_markdown.content = anvil.server.call('get_doc_markdown', self.document_chooser.selected_value)
    self.document_render_button.selected = False

  def document_table_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    if self.document_chooser.selected_value != None:
      self.document_markdown.content = anvil.server.call('get_doc_table', self.document_chooser.selected_value)
    self.document_table_button.selected = False

  def document_statistics_button_clicked(self, **event_args):
    """This method is called when this radio button is selected"""
    if self.document_chooser.selected_value != None:
      self.document_markdown.content = anvil.server.call('get_doc_statistics', self.document_chooser.selected_value)
    self.document_statistics_button.selected = False

  def document_chooser_change(self, **event_args):
    """This method is called when an item is selected"""
    pass

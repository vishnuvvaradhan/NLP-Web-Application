from ._anvil_designer import Build_Corpus_FormTemplate
from anvil import *
import anvil.server

class Build_Corpus_Form(Build_Corpus_FormTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)  

  def add_file_button_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    anvil.server.call('load_file', file.name, file)

  def add_document_button_click(self, **event_args):
    """This method is called when the button is clicked"""
    anvil.server.call('add_document', self.add_document_area.text)
    self.build_corpus_notes.content = anvil.server.call('get_document_ids')

  def clear_corpus_button_click(self, **event_args):
    """This method is called when the button is clicked"""
    anvil.server.call('clear')

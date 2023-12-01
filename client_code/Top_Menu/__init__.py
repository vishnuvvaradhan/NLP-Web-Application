from ._anvil_designer import Top_MenuTemplate
from anvil import *
import anvil.server

class Top_Menu(Top_MenuTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run before the form opens.

  def functionality_chooser_change(self, **event_args):
    """This method is called when an item is selected"""
    if self.functionality_chooser.selected_value == 'Build Corpus':
      open_form('Build_Corpus_Form', corpus="an_argument")
    elif self.functionality_chooser.selected_value == 'Analyze Corpus':
      open_form('Analyze_Corpus_Form', corpus="an argument")
    elif self.functionality_chooser.selected_value == 'Analyze Document':
      open_form('Analyze_Document_Form', corpus="an_argument")
    elif self.functionality_chooser.selected_value == 'Analyze Topics':
      open_form('Analyze_Topics_Form', corpus="an_argument")
    self.functionality_chooser.selected_value == self.functionality_chooser.items[0]

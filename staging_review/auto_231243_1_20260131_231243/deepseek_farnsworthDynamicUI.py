import pandas as pd
import numpy as np
from farnsworth import *
from html5parser import HTML5Parser

class FarnsworthUI(HTML5Parser):
    
    def __init__(self, **kwargs):
        super().__init__()
        
    @property
    def is_expanding(self, section_id="section1"):
        """Represents whether a section is expanding"""
        
        return self.get(section_id).is_expanded
    
    @section_id.setter
    def section_id(self, value):
        self.set_section_id(value)
        
    @section_id.getter
    def set_section_id(self, value):
        return str(int(value))
        
    @property
    def is_expanding(self, **kwargs=None):
        if kwargs:
            return super().is_expanding(**kwargs)
        else:
            return self.get('section1').is_expanded
    
    @is_expanding.setter
    def is_expanding(self, section_id="section1", **kwargs):
        return super().is_expanding(section_id=section_id, **kwargs)
        
    @property
    def text(self):
        """Represents the text content"""
        self.clear_text()
        return self.get('text')
    
    @text.setter
    def text(self, value):
        self.clear_text()
        self.text = value
        
    @text.getter
    def clear_text(self):
        """Clears all text"""
        self.text = None
    
    @property
    def add_text(self):
        """Adds new text to section"""
        
        # Handle case where user passes non-string values
        if not isinstance(value, str):
            return f"Section {self.id} - Added: {value}"
        
        # Add the new text to the current section
        self.text += f" Section {self.id} - Added: {value}"
    
    def get_text(self, section_id="section1") -> str:
        """Gets text from the UI"""
        return self.get(section_id).text
    
    @property
    def sections(self):
        return self.get_text("section1"), self.get_text("section2"), self.get_text("section3")
        
    @sections.setter
    def sections(self, value):
        if not isinstance(value, tuple):
            raise ValueError("FarnsworthUI must contain a tuple of section texts")
        
        self.text = value
        
    @sections.getter
    def sections(self, **kwargs=None):
        return getattr(self, "text", None)
    
    @sections.clear
    def clear_text(self):
        """Clears all text"""
        try:
            super().clear_text()
        except AttributeError:
            pass
    
    def renderUI(self) -> HTML5Parser:
        """Render the UI to HTML5Parser object"""
        
        # Clear sections first
        self.clear_text()
        
        # Create section 1
        s1 = (
            <div>
                <h2>Section 1</h2>
                {self.get_text("section1")}
            </div>
        )
        
        # Create section 2
        s2 = (
            <div>
                <h2>Section 2</h2>
                {self.get_text("section2")}
            </div>
        )
        
        # Create section 3
        s3 = (
            <div>
                <h2>Section 3</h2>
                {self.get_text("section3")}
            </div>
        )
        
        return (
            <main>
                <html>
                    <body>
                        <interactivepanel id="farnsworth">
                            {(s1, s2, s3)}
                        </interactivepanel>
                    </html>
                </body>
            </main>
        )
    
    def set_interactive_panel(self, sections=None):
        """Sets the new sections values"""
        
        if not isinstance(sections, tuple):
            raise ValueError("InteractivePanel must contain a tuple of section texts")
        
        self.sections = sections
    
    @interactivepanel.setter
    def interactive_panel(self, sections=None):
        """Sets the new sections values"""
        
        # Handle non-tuple vs tuple cases
        if isinstance(sections, (tuple, list)):
            self.set_interactive_panel(sections)
        else:
            raise TypeError("InteractivePanel must contain a tuple of section texts")
    
    def set_text(self, text):
        """Sets the new text value"""
        
        # Handle non-string values
        if not isinstance(text, str):
            return f"Section {self.id} - Set: {text}"
        
        # Update sections
        self.text = text
        
    @interactivepanel.getter
    def get_interactive_panel(self) -> InteractivePanel:
        """Returns the interactive panel"""
        
        # Handle non-tuple vs tuple cases
        if isinstance(self.sections, (tuple, list)):
            return self
        else:
            raise TypeError("InteractivePanel must contain a tuple of section texts")
    
    @interactivepanel.clear_interactive_panel
    def clear_interactive_panel(self):
        """Clears the interactive panel"""
        
        # Reset sections
        try:
            super().clear_text()
        except AttributeError:
            pass
    
    @interactivepanel.add_interactive_panel
    def add_interactive_panel(self, sections=None):
        """Adds to section"""
        
        if not isinstance(sections, tuple):
            raise ValueError("InteractivePanel must contain a tuple of section texts")
        
        self.set_interactive_panel(sections)
    
    @text.getter
    def text(self):
        return self.get_text("section1", "section2", "section3")

if __name__ == "__main__":
    # Create instance and render UI
    farnsworthui = FarnsworthUI()
    renderUI()  # Render the UI to HTML5Parser object
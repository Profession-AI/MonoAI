import xml.etree.ElementTree as ET

class PromptParser:
    def parse(self, prompt_id: str):
        prompt_file = open(prompt_id+".prompt")
        prompt_text = prompt_file.read()        
        if self._very_dumb_xml_check(prompt_text):
            return self._parse_xml(prompt_text)
        else:
            return prompt_text

    def _very_dumb_xml_check(self, text: str):
        return text[0]=="<" and text[-1]==">"
    
    def _parse_xml(self, text: str):
        root = ET.fromstring(text)
        if root.tag == "promptchain":
            return [prompt.text for prompt in root.findall("prompt")]
        elif root.tag == "iterativeprompt":
            return root.find("prompt").text, root.find("prompt_memory").text
        else:
            return root.find("prompt").text


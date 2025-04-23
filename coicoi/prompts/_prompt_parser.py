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
            return [prompt.text for prompt in root.findall("prompt")], root.get("response_type")
        elif root.tag == "iterativeprompt":
            return root.find("prompt").text, root.find("prompt_memory").text
        else:
            return root.text, self._type_from_str(root.get("response_type"))

    def _type_from_str(self, type_str: str):
        
        if type_str is None:
            return None
        
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "None": type(None)
        }
        
        assert type_str in type_mapping, f"Type {type_str} not found in type_mapping"
        return type_mapping[type_str]


if __name__ == "__main__":
    parser = PromptParser()
    print(parser.parse("test"))

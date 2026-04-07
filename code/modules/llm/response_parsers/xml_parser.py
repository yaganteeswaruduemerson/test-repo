import xml.etree.ElementTree as ET
import re
    
class XmlResponse:

    @staticmethod
    def parse_xml(xml, opening_tag):
        root = ET.fromstring(xml)
        tags = list(set([e.tag for e in root.iter()]) - {opening_tag})
        if len(tags) == 0:
            if root.text and root.text.startswith('<![CDATA[') and root.text.endswith(']]>'):
                return root.text[9:-3]  # Remove CDATA tags
            else:
                return root.text
        else:
            data = {}
            for tag in tags:
                try:
                    data[tag] = root.find(tag).text
                except:
                    continue
            return data

    @staticmethod
    def extract_response(text, opening_tag, closing_tag):
        xml_pattern = fr"{opening_tag}.*?{closing_tag}"
        isvalid = False
        xml_response = None
        try:
            xml = re.findall(xml_pattern, text, re.DOTALL)[0]
            xml_response = XmlResponse.parse_xml(xml, opening_tag[1:-1])
            isvalid = True
        except re.error as e:
            print(f"Error compiling regular expression: {e}")
        return xml_response, isvalid

    @staticmethod
    def extract_multiple(text, opening_tag, closing_tag) -> tuple[list, bool]:
        xml_pattern = fr"{opening_tag}.*?{closing_tag}"
        isvalid = False
        xml_response = []
        try:
            xml = re.findall(xml_pattern, text, re.DOTALL)
            for x in xml:
                xml_response.append(XmlResponse.parse_xml(x, opening_tag[1:-1]))
            isvalid = True
        except re.error as e:
            print(f"Error compiling regular expression: {e}")
        return xml_response, isvalid

import os
import os.path
import xml.etree.ElementTree as ET
import json
import re
import datetime

def xldate_to_datetime(xldate):
    return datetime.datetime(1899, 12, 30) + datetime.timedelta(days=xldate)

def parseDocument(sourcePath, fileName):

    docDir = os.path.join(sourcePath, "ccd")
    tree = ET.parse(docDir + "/"+ fileName + '.xml')

    root = tree.getroot()
    section = root.findall('{http://www.cicero.team/schema/ccd-20191205}section')[0]
    body = section.findall('{http://www.cicero.team/schema/ccd-20191205}body')[0] 

    textToReturn = ""
    start = 0
    attrSection=0

    for each in body:
        if start == 0:
            if "DOCUMENT ATTRIBUTES SECTION: END" in each.text:
                start = 1

        elif start == 1:
            if each.text != None:
                textToReturn = textToReturn + str(each.text)

    textToReturn = [textToReturn]
    return textToReturn


#parses xml and json at a paragraph level from existing xml and json file
def parseParagraphs(sourcePath, fileName):
    #first get the extents of all the sensitive paragraphs, i.e look at each json entry and gather the "start para" of each exemption into a list.
    sensDir = os.path.join(sourcePath, "json")
    with open(sensDir + "/" + fileName + '.json') as json_file:
        jsondata = json.load(json_file)

    sensitiveParas = []
    for annotation in jsondata:
        startP = int(annotation['extents'][0]['startPara'])
        endP = int(annotation['extents'][0]['endPara'])
        paras = list(range(int(startP), (int(endP) +  1)))
        sensitiveParas = sensitiveParas + paras

    sensitiveIndexes = list(dict.fromkeys(sensitiveParas))

    #then we want to parse out each paragraph and decide if any of the indexes for the paragraph are in the sensitiveIndexes list
    docDir = os.path.join(sourcePath, 'ccd')
    tree = ET.parse(docDir + "/" + fileName + '.xml')
    root = tree.getroot()
    section = root.findall('{http://www.cicero.team/schema/ccd-20191205}section')[0]
    body = section.findall('{http://www.cicero.team/schema/ccd-20191205}body')[0] 


    paragraphs = []
    created = []
    i = 0
    start = 0
    attrSection = 0
    labels = []
    butes = []
    paragraph = ""

    doc_created = None
    for each in body:
        if attrSection == 0:
            if "DOCUMENT ATTRIBUTES SECTION: START" in each.text:
                attrSection = 1
        elif attrSection == 1:
            if "Document Date:" in each.text:
                doc_created=xldate_to_datetime(int(str(each.text).split(":")[-1].strip()))
                attrSection = -1

        if start == 0:
            if "DOCUMENT ATTRIBUTES SECTION: END" in each.text:
                start = 1
                attrSection = -1

        elif start == 1:
            if each.text == None or str(each.text).isspace():
                if paragraph != "":
                    tba = re.sub(' +', ' ', paragraph)
                    paragraphs = paragraphs + [tba]
                    check = check =  any(item in butes for item in sensitiveIndexes)
                    if check:
                        labels = labels + [1]
                    else:
                        labels = labels + [0]

                    created.append(doc_created)

                    butes = []
                    paragraph = ""

            
            else:
                paragraph = paragraph + each.text
                butes = butes + [int(each.attrib['index'])]
    

    return paragraphs, labels, doc_created


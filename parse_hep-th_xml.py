import cPickle
from lxml import etree

tree = etree.parse(sys.argv[1])

root = tree.getroot()

objects = root[0]
links = root[1]
attributes = root[2]

link_types = [ x for x in attributes if x.attrib["NAME"] == "linktype" ][0]
object_types = [ x for x in attributes if x.attrib["NAME"] == "objecttype" ][0]

paper_ids = [ x.attrib["ITEM-ID"] for x in object_types if x[0].text == "Paper" ]
author_ids = [ x.attrib["ITEM-ID"] for x in object_types if x[0].text == "Author" ]
authored_links_ids = set([ link.attrib["ITEM-ID"] for link in link_types if link[0].text == "Authored" ])

# O1-ID is author id, O2-ID is paper id
paper_author_links = [ link for link in links if link.attrib["ID"] in authored_links_ids ]

paper_arXiv_ids = [ x for x in attributes if x.attrib["NAME"] == "paper_id" ][0]
paper_paper_id_map = {x.attrib["ITEM-ID"] : x[0].text for x in paper_arXiv_ids}

paper_author_map = { paper_paper_id_map[paper] : [] for paper in paper_ids }
for link in paper_author_links:
  paper_author_map[paper_paper_id_map[link.attrib["O2-ID"]]].append(link.attrib["O1-ID"])

with open("paper_author_map.pkl", 'w') as f:
  cPickle.dumps(paper_author_map, f)

#!/usr/bin/python
# coding=utf-8
import sys
from xml.dom.minidom import parse
import xml.dom.minidom


def get_tagname(node, tagname):
    """
    use '\r\n' to join multi items
    use '|||' to join ['ref-source', 'ref-name', 'ref-url']

    :param node: xml node
    :param tagname: name of tag
    :return: string of tagname
    """
    tagNode = ''
    try:
        if tagname == 'vuln-software-list':
            tagNode = '\r\n'.join([_p.childNodes[0].data for _p in node.getElementsByTagName('product')])
        elif tagname == 'refs':
            _refkeys = ['ref-source', 'ref-name', 'ref-url']
            tagNodes = node.getElementsByTagName('ref')
            tagNode = '\r\n'.join(map(lambda _tagNode: '|||'.join(
                map(lambda _refkey: _tagNode.getElementsByTagName(_refkey)[0].childNodes[0].data, _refkeys)), tagNodes))
        else:
            tagNode = node.getElementsByTagName(tagname)[0].childNodes[0].data
    except IndexError:
        pass
    return tagNode


def parse_url(filename):
    """parse xml file

    :param filename: xml file
    :return: list of xml items, each item is dict type
    """
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    entries = collection.getElementsByTagName('entry')
    _vul_keys = ['name', 'vuln-id', 'published', 'modified', 'source', 'severity', 'vuln-type', 'vuln-software-list',
                 'vuln-descript', 'cve-id', 'bugtraq-id', 'vuln-solution', 'refs']

    return [{_vul_key: get_tagname(entry, _vul_key) for _vul_key in _vul_keys} for entry in entries]


if __name__ == '__main__':
    data = parse_url('2000.xml')

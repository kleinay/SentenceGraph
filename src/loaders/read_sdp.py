"""
usage: read_sdp.py --in=SDP-input-file 

Handle SDP data format. Contains functions for: 
* load form file
* convert to a JSON object with all information
* convert to SentenceGraph (wrapper of networkx.DiGraph)
* visualize using Brat 

the JSON data-structure for a SdpSentence:
 {  "sentence" : <the sentence>,
    "sentence_id" : <sent_id>
    "tokens" : { token-index : SdpToken}
  SdpToken is a dict representation of the data of the token.   

"""

import logging, os.path, io
logging.basicConfig(level=logging.INFO)

COLUMNS_TEST = "id, form, lemma, pos".split(", ")
COLUMNS_TRAIN = COLUMNS_TEST + "top, pred, frame".split(", ")


def read_sdp_data_to_SentenceGraphs(sdpFileName):
    """ return a list of SentenceGraphs for each sentence in the SDP data"""
    return [getSentenceGraph(sdpSentenceJson) for sdpSentenceJson in read_sdp_data_to_JSONs(sdpFileName)]

def read_sdp_data_to_JSONs(sdpFileName):
    """ return a list of a JSON objects (python dict) in the format of SdpSentence """
    SdpSentences = []
    with io.open(sdpFileName, "r", encoding="utf-8") as f:
        # skip first line in file in case it starts with #SDP
        firstLine=next(f)
        if firstLine.split()[0]!="#SDP":
            block=firstLine # otherwise- keep first line in block
        else:
            block = ""
        for line in f:
            if not line.strip(): # empty line - declaring a new block
                # wrap and parse last block
                SdpSentence = read_SdpSentence(block)
                SdpSentences.append(SdpSentence)
                # logging.debug("collected annotation for sentence: " + SdpSentence["sentence"])
                block = ""

            else:
                block += line
    return SdpSentences


def read_SdpSentence(block):
    tokens_data = {}
    # first line in block is sentence id
    block_lines = [line.strip() for line in block.splitlines()]
    sentence_id = block_lines[0].lstrip("#")
    for line in block_lines[1:]:    # each line stands for a token
        data = token_line_to_data(line)
        id = int(data["id"])     # id is index of token (starting at 1)
        tokens_data[id] = data
    # retrieve sentence for FORM field
    tokSent = [tokens_data[i]["form"] for i in range(1, len(tokens_data)+1)]
    return {"sentence" : ' '.join(tokSent),
            "tokSent" : tokSent,
            "sentence_id" : sentence_id,
            "tokens" : tokens_data,
            "raw": block}


def token_line_to_data(line_str):
    """
    :param line_str: a line for a token  
    :return: SdpToken: a dict representation for the data of the token.
    "arg1", "arg2" and so on are the keys for columns len(COLUMNS_TRAIN) +1, len(COLUMNS_TRAIN) +2 and so on.
    """
    def column_key_iterator():
        # the first columns are constants
        for key in COLUMNS_TRAIN:
            yield key
        # after these, iterating "argOf#1" "argOf#2" etc
        i=0
        while True:
            i+=1
            yield "argOf#" + str(i)

    return dict((key, data) for key,data in zip(column_key_iterator() ,line_str.split()))


def getSentenceGraph(SdpSentence):
    """  
    :param SdpSentence:  the "inhouse" JSON representation of an SDP representation for a sentence 
    :return: a SentenceGraph object with the graph of the sentence. 
    """
    # see long comment below.
    def getPredicateId2tokenId(SdpSentence):
        predicatesTokenIds = [ti
                              for ti,td in SdpSentence["tokens"].iteritems()
                              if td['pred']=='+']
        return {predicateIndex : int(tokenId) for predicateIndex, tokenId in enumerate(predicatesTokenIds, start=1)   }

    predicateId2tokenId = getPredicateId2tokenId(SdpSentence)

    from SentenceGraph import SentenceGraph
    g = SentenceGraph(SdpSentence["tokSent"], word_indexed=True)
    # turn raw JSON representation into networkx graph using SentenceGraph
    # go through all tokens, and create edges for their "argOf" attributes
    for ti, td in SdpSentence["tokens"].iteritems():
        """
        argOfRelations is the raw data in the original SDP file format.
         Note that the head-indexes in this format are not the token index (or id), but are 
         the **predicate** index, i.e. the relative index when only taking tokens that are 
         marked as a predicate (at the "pred" column).
         For this end, we use a 'predicateId2tokenId' dict.
        """
        argOfRelations = {int(k.lstrip("argOf#")):v for k,v in td.iteritems()
                          if k.startswith("argOf#") and v != "_"}
        for head_predicate_idx, label in argOfRelations.iteritems():
            # retrieve head's token index from it's predicate-index
            head_idx = predicateId2tokenId[head_predicate_idx]
            g.add_edge(head_idx-1, int(ti)-1, label)
    return g


# helper functions (after loading all data)

# create html visualization for all sentences in data at some output-directory
def visualize_all(data, output_dir, output_prefix):
    for s in data:
        g = getSentenceGraph(s)
        out_fn = os.path.join(output_dir, output_prefix + str(s["sentence_id"]) + ".html")
        g.visualize(out_fn)

# return data of sentences that validate some condition.
# @arg filterFunction is a (sentence-str => Boolean) function.
def get_data_of(filterFunction, data):
    return [s for s in data if filterFunction(s["sentence"])]


# main - for interactive exploration of SDP data
if __name__ == "__main__":
    # Parse arguments
    from utils import *
    from docopt import docopt
    args = docopt(__doc__)
    inp_fn = args["--in"]
    # load data to JSONs
    data = read_sdp_data_to_JSONs(inp_fn)
    graphs = [getSentenceGraph(sdpJson) for sdpJson in data[:20]]

    """
    Example usage (in interactive mode):
    >>> s = data[0] # s is data of first sentence, at JSON format
    >>> g = getSentenceGraph(s)
    >>> g.visualize("sdp_sentence.html")
    OR:
    >>> graphs = read_sdp_data_to_SentenceGraphs(sdpFileName)   # load SDP data directly to list of SentenceGraphs 
    >>> g = graphs[0]
    
    # get underlying networkx.DiGraph:
    >>> nx_graph = g.graph
    """

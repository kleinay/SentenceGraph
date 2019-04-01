import networkx as nx
from networkx.readwrite import json_graph
import utils
#from utils import *
import json, logging, sys
logging.basicConfig(level=logging.DEBUG)

"""
For brat visualization - requires installation of brat-visualizer and downloading brat 
(by the brat-visualizer/download_brat.sh script)
"""
default_brat_root_location = "lib/brat-visualizer/brat/brat-v1.3_Crunchy_Frog"

class SentenceGraph:
    """
    Simple wrapper of networkx DiGraph for a sentence level bilexical graph representation.
    Nodes are (start-index, end-index) tuples (indexed by token or by character) 
    """

    def __init__(self, sentence, word_indexed=True, **kwargs):
        """
        Initialize with a space separated sentence string, or a list of tokens (pre-tokenized sentence).
        @:param sentence: str or list of str (tokenized).
            if str - tokenization is done simply by space (str.split()) 
            if list- sentence string is determined by joining tokens over a space (' '.join(list)) 
        @:param word_indexed: boolean, indicating whether the indices used by underlying Graph 
        (and by add_edge function) are indices of words, referring to tokenized sentence (True), 
        or indices of character, referring to sentence string (False).
        """
        assert type(sentence) in [str, unicode, unicode, list], "sentence must be a string or a list of tokens"
        # self.tokSent is the tokenized sentence (used for word-indices)
        # self.sent is the sentence surface string (used for character indices)
        if type(sentence) in [str, unicode]:
            self.tokSent = sentence.split()
            self.sent = sentence
        else:
            self.tokSent = sentence
            self.sent = " ".join(sentence)
        # create token2index mappings
        i2t=list(enumerate(self.tokSent))

        self.tok_occ2index = utils.getElementOccurenceMapping(self.tokSent)

        # create empty graph
        self.graph = nx.DiGraph()
        self.word_indexed = word_indexed
        # set default project-relative brat location
        self.setBratLocation(default_brat_root_location)
        # add kwargs as attributes
        self.__dict__.update(kwargs)

    @staticmethod
    def from_repr(repr_str, keys = ["label", "rule"], sentence=None, **kwargs):
        """
        Inverse of __repr__ - Creates a new SentenceGraph from a textual representations.
        The representation is in the format (example):
        "Mr. Ittleson is executive , special projects , at CIT Group Holdings Inc. .
         (u'Mr.', u'projects', u'What is Mr.?#9', 'maxCover')
         (u'Mr.', u'CIT Group Holdings Inc.', u'Where is Mr. at?#14', 'singleWord')
         (u'Ittleson', u'projects', u'What is Ittleson to do?#13', 'posForPairs')
         (u'Ittleson', u'CIT Group Holdings Inc.', u'Where was Ittleson?#10', 'singleWord')"

        :param repr_str: 
        :param keys: specify the keys of the 3rd, 4th, ... elements in each edge-tuple.
        :param sentence: if not specified (default), read first line of repr_str as sentence.
        Otherwise, take sentence from argument and read repr_str as textual list of edges.
        :param kwargs: 
        :return: a new SentenceGraph object
        """
        # retrieve sentence and seperate it from rest of representation
        repr_str = repr_str.strip()
        if not sentence:
            sentence = repr_str.splitlines()[0]
            repr_str = '\n'.join(repr_str.splitlines()[1:])
        edgeList = [utils.str2obj(line.strip())
                    for line in repr_str.splitlines()]
        return SentenceGraph.from_edges(sentence, edgeList, edge_keys=keys, **kwargs)

    @staticmethod
    def from_edges(sentence, edgeList, word_indexed = True, edge_keys = ["label"], **kwargs):
        """
        Creates a new SentenceGraph from list of edges.
        :param sentence: the ref sentence (string or list of tokens)
        :param edgeList: a list of tuples representing edges. an Edge can be 2-tuple (u,v) or 3-tuple (u,v,label).
            Nodes (such as u & v) are 2-tuple (start,end) that corresponds to a span in the sentence.
            start and end indices are either token-indices (word_indexed=True, default) or character-indices (word_indexed=False)
        :param word_indexed:
        :return: a new SentenceGraph object
        """
        # g = SentenceGraph(sentence, word_indexed, **kwargs)
        # if edgeList and len(edgeList[0])==2:
        #     labeledEdges = False
        # else:
        #     labeledEdges = True
        # if labeledEdges:
        #     for edge in edgeList:
        #         u, v, label = edge[:3]
        #         g.add_edge(u,v,label)
        # else:
        #     for u,v in edgeList:
        #         g.add_edge(u,v,"")

        g = SentenceGraph(sentence, word_indexed, **kwargs)
        # must have a label property somewhere; if not specified in keys, assumed to be the third
        if "label" not in edge_keys:
            edge_keys = ["label"] + edge_keys
        for edge in edgeList:
            u, v = edge[:2]
            edge_properties = dict(zip(edge_keys, edge[2:]))
            g.add_edge(u,v,**edge_properties)

        return g

    def add_edge(self, u, v, label, **kwargs):
        """ maps to specific add_edge functions by the type of u & v. Underling nodes are always indices. """
        kwargs["label"]=label   # label is mandatory
        assert type(u) in [str, unicode, int, list, tuple], "Node can be either string, or list of token strings, or tuple of indices, or a single index"
        if type(u) is int:
            self.add_edge_by_indices((u,1+u), (v,1+v), **kwargs)
        elif type(u) in [list, tuple] and len(u)==2 and type(u[0]) is int and type(u[1]) is int:
            self.add_edge_by_indices(u,v,**kwargs)
        elif type(u) in [str, unicode] \
                or type(u) in [list, tuple] and len(u)>=1 and type(u[0]) in [str, unicode]:
            self.add_edge_by_token_lookup(u,v,**kwargs)
        else:
            raise Exception("nodes are not of suitable type")

    def add_edge_by_indices(self, u, v, **kwargs):
        """
        Add a directed edge u -> v with a given label
        u and v should be 2-tuple of word indices indicating start and end (exclusive!)
        i.e., u = (u_start, u_end), v = (v_start, v_end),
        """
        self.graph.add_edge(u, v)
        for key in kwargs:
            self.graph[u][v][key] = kwargs[key]

    def add_edge_by_token_lookup(self, uStr, vStr, **kwargs):
        """ uStr and vStr can be either strings (taken from sentence- 1st occurrence is selected)
            or list/tuple of 2 tokens (start-span-token, end-span-token),
            or (token-string, #-of-occurrence) tuple,
                    [e.g. ("man", 2) refers to 2nd occurrence of "man" in sentence] """
        def extractArgIndices(uStr):
            uStrFirstOcc, uStrLastOcc = 1,1 # by default, first occurrence is referred
            if type(uStr) in [str, unicode]:
                # check if uStr is a single token
                if uStr in self.tokSent:
                    uStrFirst, uStrLast = uStr, uStr
                else:
                    # if not, allow for span annotation (shouldn't be used in gold QASemDep graphs)
                    uStrFirst, uStrLast = uStr.split()[0], uStr.split()[-1]
            elif len(uStr)==2 and type(uStr[0]) in [str, unicode] and type(uStr[1]) is int: # (word, #-occurrence)
                # (token-string, #-of-occurrence) tuple
                uStrFirst, uStrLast = uStr[0], uStr[0]
                uStrFirstOcc, uStrLastOcc = uStr[1], uStr[1]
            else:
                uStrFirst, uStrLast = uStr[0], uStr[-1]
            assert (uStrFirst,uStrFirstOcc) in self.tok_occ2index, "first token "+uStrFirst+" for node was not found in tokenized sentence"
            assert (uStrLast, uStrLastOcc) in self.tok_occ2index, "last token for node was not found in tokenized sentence"
            # add 1 to last-word index because span is defined in SentenceGraph as (inclusive-index, exclusive-index)
            return (self.tok_occ2index[uStrFirst, uStrFirstOcc], 1+self.tok_occ2index[uStrLast, uStrLastOcc])

        u = extractArgIndices(uStr)
        v = extractArgIndices(vStr)
        self.add_edge_by_indices(u, v, **kwargs)

    def setBratLocation(self, brat_location):
        self.brat_location = unicode(brat_location)

    def getSpan(self, (start_index, end_index)):
        """
        return the text in the sentence that corresponds to this span (as String).
        Consider the flag self.word_indexed to decide whether to take from tokSent (True) or from sent (False)
        """
        if self.word_indexed:
            return " ".join(self.tokSent[start_index:end_index])
        else:
            return self.sent[start_index:end_index]

    """
    Structural Characteristics of graph
    """
    def __len__(self):
        """ Return the number of edges in the graph. """
        return len(self.graph.edges)

    def countNonProjectivity(self):
        """
        :return: number of "Crossings" between edges in the graph 
        """
        def is_between(x, (u,v)):
            # is x between u and v? (if x equals to u or v, return False)
            u,v = sorted((u,v))
            return u<x and x<v
        def is_not_between(x, (u,v)):
            # is x explicitly outside of (u,v)? (if x equals to u or v, return False)
            u,v = sorted((u,v))
            return x<u or x>v
        def is_crossing(edge1, edge2):
            # retrieve relevant indices u1,v1,u2,v2 (Ints)
            ((u1,_), (v1,_)), ((u2,_), (v2,_)) = sorted((edge1, edge2))
            return (is_between(u1, (u2,v2)) and is_not_between(v1, (u2,v2))) \
                or (is_between(v1, (u2,v2)) and is_not_between(u1, (u2,v2)))
        # count number of crossing edge-pairs
        crossing_edge_pairs = [(e1,e2)
                               for e1, e2 in utils.Span.all_pairs(list(self.graph.edges))
                               if is_crossing(e1, e2)]
        return len(crossing_edge_pairs)

    def is_connected(self):
        return nx.is_connected(self.graph.to_undirected())

    def is_tree(self):
        return nx.is_tree(self.graph.to_undirected())

    def diameter(self):
        return nx.diameter(self.graph.to_undirected())


    """
    word-indexed to char-indexed functionality
    """

    def word2charSpan(sg, wordIndex):
        """
        Return (begin-char-idx, end-char-idx), 
        where end-char-idx is exclusive, i.e. sent[end-idx] is the space after the word.
        :param sg: SentenceGraph
        """
        char_start = wordIndex + sum(map(len, sg.tokSent[: wordIndex]))
        char_end = char_start + len(sg.tokSent[wordIndex])  # exclusive
        if sg.sent[char_start:char_end] != sg.tokSent[wordIndex]:
            logging.warning("Mismatch between SG.sent and SG.tokSent. Verify both with same encoding.")
        return (char_start, char_end)

    def wordSpan2charSpan(sg, wordSpan):
        # for handling SentenceGraph graph nodes, which are (word-)spans
        word_start, word_end = wordSpan  # wordSpan is exclusive- tokSent[end] is not part of the span
        return (sg.word2charSpan(word_start)[0],  # first char of first word
                sg.word2charSpan(word_end - 1)[1])  # last char of last included word

    def as_char_indexed(self):
        """return a copy of the SentenceGraph with word_indexed=False"""
        if not self.word_indexed:
            return self
        # iterate all edges and copy them to another
        newSG = SentenceGraph(self.tokSent, word_indexed=False)
        for edge_ids, edge_data in self.graph.edges.iteritems():
            # edge_ids are a word-indixes span (2-tuple). We want to convert it to char-indices
            char_start = self.wordSpan2charSpan(edge_ids[0])
            char_end = self.wordSpan2charSpan(edge_ids[1])
            newSG.graph.add_edge(char_start, char_end, **edge_data)
        return newSG


    def diff(self,
             otherSentenceGraph,
             directed=False,
             edgeMatchCriteria = lambda e1,e2: e1==e2):
        """
        Return a SentenceGraph which contains the difference between self and other, self - other,
         i.e., edges in self but not in other.
        If self is predicted, and other is gold, diff is the False-Positive (=precision) Errors.
        If self is gold, and other is predicted, diff is the False-Negative (=recall) Errors.
        """
        otherGraph = otherSentenceGraph.graph
        selfNodeIntersect=self.graph.copy()
        if not directed:
            otherGraph = otherGraph.to_undirected()

        # return a SentenceGraph wrapper of the difference
        diffSG = SentenceGraph(self.tokSent, word_indexed=self.word_indexed)
        diffSG.graph = selfNodeIntersect.copy()
        for edge in selfNodeIntersect.edges:
            # find a matching edge in other graph
            matching_edges = [other_edge for other_edge in otherGraph.edges if edgeMatchCriteria(edge, other_edge)]
            if matching_edges:
                diffSG.graph.remove_edge(*edge)
        return diffSG

    def match(self, otherSentenceGraph, directed=False, edgeMatchCriteria = lambda e1,e2: e1==e2):
        """ Return a SentenceGraph containing the intersection of two graphs. """
        assert self.tokSent == otherSentenceGraph.tokSent, "Tokenized Sentence must be equivalent to find match"

        otherGraph = otherSentenceGraph.graph
        selfNodeIntersect = self.graph.copy()
        if not directed:
            otherGraph = otherGraph.to_undirected()

        # return a SentenceGraph wrapper of the difference
        matchSG = SentenceGraph(self.tokSent, word_indexed=self.word_indexed)
        matchSG.graph = selfNodeIntersect.copy()
        for edge in selfNodeIntersect.edges:
            # find a matching edge in other graph
            matching_edges = [other_edge for other_edge in otherGraph.edges if edgeMatchCriteria(edge, other_edge)]
            if not matching_edges:
                matchSG.graph.remove_edge(*edge)
        return matchSG

    def count_strict_match(self, goldSentenceGraph, directed=False):
        """ Return (#self edges, #gold edges, #matched edges)  
            (to be used as input bu utils.compute_F1PrR)
        """
        predictedSentenceGraph = self

        # verify sentence matches
        if self.sent != goldSentenceGraph.sent:
            #logging.warning("Comparing graphs with different sentences.")
            logging.warning("Comparing graphs with different sentences. \n S1: {}\n S2: {} ".format(
                self.sent.encode("utf-8"), goldSentenceGraph.sent.encode("utf-8")))
            # raise Exception("Comparing graphs with different sentences")
        elif self.tokSent != goldSentenceGraph.tokSent:
            mismatch = utils.firstMismatch(self.tokSent, goldSentenceGraph.tokSent)
            logging.warning("Comparing graphs with tokenization mismatch: " + str(mismatch))
            # still, try comparing

        from networkx.algorithms import difference, intersection
        gold = goldSentenceGraph.graph
        predicted = predictedSentenceGraph.graph.subgraph(gold.nodes).copy()
        predicted.add_nodes_from(gold.nodes)    # so that both graphs will have same nodes

        # for undirected comparison - change DiGraph to undirected Graphs
        if not directed:
            gold = gold.to_undirected()
            predicted = predicted.to_undirected()
        falsePositiveGraph = difference(predicted, gold)
        falseNegativeGraph = difference(gold, predicted)
        matchGraph = intersection(gold, predicted)      # true positive

        return (len(predictedSentenceGraph.graph.edges),
                len(gold.edges),
                len(matchGraph.edges))

    @staticmethod
    def get_edge_match_criteria(directed, span_overlap):
        """       
        :param directed: 
        :param span_overlap: 
        :return: a function edge_match_criteria(edge1, edge2) that return True for matching edges
        """
        # define nodeMatchCriteria for comparison
        if span_overlap:
            nodeMatchCriteria = lambda node1, node2: bool(utils.Span.intersect(node1, node2))
        else:
            # if demanding strict match of nodes - use strict match (should better be on char-based graphs)
            nodeMatchCriteria = lambda node1, node2: node1==node2

        # This edge-match criteria is straightforwardly induced from the nodeMatchCriteria
        if directed:
            edgeMatchCriteria = lambda edge1, edge2: nodeMatchCriteria(edge1[0], edge2[0]) \
                                                     and nodeMatchCriteria(edge1[1], edge2[1])
        else:
            edgeMatchCriteria = lambda edge1, edge2: (nodeMatchCriteria(edge1[0], edge2[0])
                                                      and nodeMatchCriteria(edge1[1], edge2[1])) or \
                                                     (nodeMatchCriteria(edge1[0], edge2[1])
                                                      and nodeMatchCriteria(edge1[1], edge2[0]))
        return edgeMatchCriteria


    def compare(self, goldSentenceGraph, directed=False):
        """ Return f1, precision, recall of comparing self to gold graph. use strict edge match criteria. """
        return utils.compute_F1PrR(*self.count_strict_match(goldSentenceGraph, directed))

    def count_relaxed_match(self, goldSentenceGraph, directed=False, span_overlap=True):
        """ 
        Return (#self edges, #gold edges, #matched edges) with a relaxed-comparison criteria between edges.
        In Further Details:
        Make a relaxed comparison between the graphs.
        Useful for comparing different formalisms, especially where the tokenization of the sentence
        may differ. 
        In the relaxed comparison, we roll back from word-indices comparison (nailed to a certain tokenization),
        and instead use char-indices comparison between nodes. 

        Additionally, we allow for a span-overlap to be considered as a match; e.g. ((0,5), (10,20)) can
         be considered as matching an edge ((0,5), (15,20)).
        """
        self_ch = self.as_char_indexed()
        gold_ch = goldSentenceGraph.as_char_indexed()

        edgeMatchCriteria = SentenceGraph.get_edge_match_criteria(directed, span_overlap)

        # define edge-ignore criteria - these edges would be ignored, i.e., not counted as mismatch.
        # (obviously they are also not counted as a match, because no self-cycles in graphs.)
        # Usage: we want to ignore edges included within a single node (span)- to overlook tokenization mismatches.
        # e.g. if "Mr. Vinken" is a single node in graph G, edge "Mr."->"Vinken" in graph H would be ignored.
        spanOfEdge = lambda edge: (sorted(utils.flatten(edge))[0], sorted(utils.flatten(edge))[-1])
        isEdgeContainedWithinNode = lambda edge, node: utils.Span(spanOfEdge(edge)).is_within(node)
        def edgeIgnoreCriteria(edge, otherGraph):
            return any(isEdgeContainedWithinNode(edge, node) for node in otherGraph.nodes)

        # compute precision and recall based on our matching criteria
        matching_edges = [edge
                          for edge in self_ch.graph.edges
                          if any(goldEdge
                                 for goldEdge in gold_ch.graph.edges
                                 if edgeMatchCriteria(edge, goldEdge))]
        self_ignored_edges = [edge
                              for edge in self_ch.graph.edges
                              if edgeIgnoreCriteria(edge, gold_ch.graph)]
        gold_ignored_edges = [edge
                              for edge in gold_ch.graph.edges
                              if edgeIgnoreCriteria(edge, self_ch.graph)]

        return (len(self_ch.graph.edges) - len(self_ignored_edges),
                len(gold_ch.graph.edges) - len(gold_ignored_edges),
                len(set(matching_edges) - set(self_ignored_edges + gold_ignored_edges)))

    def lax_compare(self, goldSentenceGraph, directed=False, span_overlap=True):
        """ Return f1, precision, recall of comparing self to gold graph. use relaxed edge match criteria. """
        return utils.compute_F1PrR(*self.count_relaxed_match(goldSentenceGraph,
                                                       directed=directed,
                                                       span_overlap=span_overlap))


    # Visualizations (textual and graphical)

    def edge_repr(self, (u,v), directed=True, withSentence=False):
        """
        Return a tuple textually representing the edge
        :param directed: When False, search the opposite-dircted edge if edge not found
        :param withSentence: whether to include the sentence string in the tuple
        """
        edge = (u,v)
        if not directed and (u,v) not in  self.graph.edges and (v,u) in self.graph.edges:
            edge = (v,u)
        if edge not in self.graph.edges:
            return None
        edge_info = self.graph.edges[edge]
        label = edge_info["label"] if "label" in edge_info else ""
        # for debug
        rule = edge_info["rule"] if "rule" in edge_info else ""
        edge_repr_tuple = (self.getSpan(u), self.getSpan(v), label, rule)
        if withSentence:
            edge_repr_tuple = (self.sent,) + edge_repr_tuple
        return edge_repr_tuple

    def view(self):
        """
        return a textual view (list of lexicalized edges) of the graph
        """
        edges_as_strings=[]
        for u,v in sorted(list(self.graph.edges)):
            edges_as_strings.append(self.edge_repr((u,v)))
        sentRep= self.sent
        edgesRep = "\n".join(unicode(e)
                             for e in edges_as_strings)
        rep = sentRep + "\n" + edgesRep
        return rep

    def __repr__(self):
        # view
        # encode back to byte sequence, so that presenting interactivelt won't fail
        return self.view().encode("utf-8")

    def visualize(self, fn, visualizer='displacy', edgeLabelProcessFunc=lambda s:s):
        """
        Visualize this structure into a brat html file.
        :param fn: output html file name
        :param visualizer:  'brat' to write a Brat visualization (html), 
                            'displacy' to write a displacy visualization (svg) 
        :param edgeLabelProcessFunc: a function to process the edge labels of the graph before visualizing. 
        the edgeLabelProcessFunc function should get a string (label) and return a string 
        """
        if visualizer=='brat':
            try:
                if "lib/brat-visualizer/src" not in sys.path:
                    sys.path.insert(0, "lib/brat-visualizer/src")
                from brat_handler import Brat
                Brat.output_brat_html(self.sent,
                                      self.graph,
                                      fn,
                                      self.brat_location,
                                      self.word_indexed)
            except ImportError:
                print("You must install brat-visualizer before using the brat visualizer.")
                print("Try running: `sh install_brat_visalizer.sh`")

        elif visualizer=='displacy':
            try:
                from displacy_handler import output_displacy_svg
                output_displacy_svg(self.tokSent,
                                    self.graph,
                                    fn,
                                    edgeLabelProcessFunc)
            except ImportError:
                print("You must install SpaCy before using the displacy visualizer.")


    # <editor-fold desc="Serialization and Deserialization">
    def asJson(self):
        return {"graph": json_graph.node_link_data(self.graph),
                "word_indexed": self.word_indexed,
                "sentence": self.sent,
                "tokenized": self.tokSent}

    def dumps(self, *args, **kwargs):
        return json.dumps(self.asJson(), *args, **kwargs)

    def dump(self, fd, *args, **kwargs):
        return json.dump(self.asJson(), fd, *args, **kwargs)

    def __str__(self):
        return self.dumps(indent=3)

    @staticmethod
    def fromJson(jObj):
        sentGraph = SentenceGraph(jObj["tokenized"], word_indexed=jObj["word_indexed"])
        sentGraph.graph = json_graph.node_link_graph(jObj["graph"])
        return sentGraph

    @staticmethod
    def load(fp, *args, **kwargs):
        """ Create a SentenceGraph object out of a serialization of it"""
        jObj = json.load(fp, *args, **kwargs)
        return SentenceGraph.fromJson(jObj)

    @staticmethod
    def loads(s, *args, **kwargs):
        """ Create a SentenceGraph object out of a serialization of it"""
        jObj = json.loads(s, *args, **kwargs)
        return SentenceGraph.fromJson(jObj)
    # </editor-fold>


"""
    ## Example usage

    # create instance with the sentence (words separated with space)
    g = SentenceGraph("the brown fox jumped over the lazy dog")

    # edges and labels
    g.add_edge((3, 4), (2, 3), "Who jumped?")
    g.add_edge((2, 3), (1, 2), "What color was the fox?")
    g.add_edge((3, 4), (7, 8), "Who did the fox jump over?")
    g.add_edge((6, 7), (7, 8), "Who was lazy?")

"""

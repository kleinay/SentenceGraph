import ast, json, io, sys, logging, itertools, numpy as np
logging.basicConfig(level=logging.INFO)

def getElementOccurenceMapping(iterable):
    """ Return a mapping (element, #-occurrence) => index in iterable.
        e.g. for the list l=["a","b","c","a","c","c","f","g"]:
            getElementOccurenceMapping(l)["a",1] = 0 ;
            getElementOccurenceMapping(l)["a",2] = 3 ;
            getElementOccurenceMapping(l)["c",1] = 2 ;
            getElementOccurenceMapping(l)["a",3] = 5 ;
    """
    from collections import Counter
    elementCounter = Counter()
    element_occ2index = {}
    for i, t in enumerate(iterable):
        elementCounter[t] += 1
        element_occ2index[t, elementCounter[t]] = i
    return element_occ2index

def loadLineSplitFiles(file_name, deserialize_block_func=lambda blockString: blockString,
                       skip_first_line=False,
                       encoding="utf-8"):
    """ Return an iterable of (deserialized) blocks of a file, where blocks are delimited by an empty line. 
       Useful for loading CoNLL or CoNLL-like annotation files.
    :param file_name: 
    :param deserialize_block_func:
        If provided, the function will yield the block deserialized into an object. 
    :return: yield deserialize_block_func(block) for block in the file
    """
    with io.open(file_name, "r", encoding=encoding) as f:
        # skip first line in file in case skip_first_line
        if skip_first_line:
            firstLine = next(f)
        block = ""
        for line in f:
            if not line.strip(): # empty line - declaring a new block
                # wrap and parse last block
                yield deserialize_block_func(block)

                # logging.debug("collected annotation for sentence: " + <sentence-string>)
                block = ""
            else:   # continue with collecting the block
                block += line


def loadJsonWithComments(json_fn, commentSymbol="#"):
    """ Load a json file that contains comments - ignore lines starting with commentSymbol """
    with open(json_fn, "r") as f:
        jsonStr = ''.join([line for line in f if not line.startswith(commentSymbol)])
    import json
    return json.loads(jsonStr)

def loadJson(json_fn):
    return json.load(open(json_fn, "r"))

def str2obj(json_like_string):
    return ast.literal_eval(json_like_string)

def saveJson(obj, outputFileName="output.json"):
    with open(outputFileName, "w") as f:
        json.dump(obj, f, indent=3, sort_keys=True)


# data (SDP or QASRL) usage utils

def getSentenceId(sdp_data, sentence_prefix_str):
    # get the SDP sentence ID for the sentence starting with sentence_prefix_str
    for sdpJson in sdp_data:
        if sdpJson['sentence'].lower().startswith(sentence_prefix_str.lower()):
            return sdpJson['sentence_id']
    return "not found"

def getSpecificSDPJsons(sdp_full_data, list_of_sentence_ids):
    id2sdpJson = { sdpJson['sentence_id'] : sdpJson for sdpJson in sdp_full_data }
    return {sent_id : id2sdpJson[sent_id] for sent_id in list_of_sentence_ids}

### Here is a version which tried to retrieve the SDP dat by sentence prefix.
### this had problems, since the SDP sentences is encoded differently than the gold-sentences, and there is mismatch
### that makes it hard to find the sentence data by it's string.
# def getSpecificSDPJsons(sdp_full_data, list_of_sentences_str):
#     import read_sdp
#     sdpJsons=[]
#     for sent in list_of_sentences_str:
#         for sdpJson in sdp_full_data:
#             if sdpJson['sentence'].lower().startswith(str(sent[:-10]).lower()):
#                 sdpJsons.append(sdpJson)
#     return sdpJsons
#
# list, dict and sets utils
def is_iterable(e):
    return '__iter__' in dir(e)

def flatten(lst, recursively=False):
    """ Flatten a list.
    if recursively=True, flattens all levels of nesting, until reaching non-iterable items 
    (strings are considered non-iterable to that matter.) 
    :returns a flatten list (a non-nested list)
    """
    if not is_iterable(lst):
        return lst
    out = []
    for element in lst:
        if is_iterable(element):
            if recursively:
                out.extend(flatten(element))
            else:
                out.extend(element)
        else:
            out.append(element)
    return out

def is_nested(lst):
    return any(is_iterable(e) for e in lst)

def power_set(lst, as_list=True):
    """ set as_list to false in order to yield the power-set """
    import itertools
    pwset_chain = itertools.chain.from_iterable(itertools.combinations(lst, r)
                                                for r in range(len(lst)+1))
    if as_list:
        return list(pwset_chain)
    else:
        return pwset_chain

def n_choose_k(lst, subset_size, as_list=True):
    """ return all subsets of lst with size #subset_size """
    generator = (subset
                 for subset in power_set(lst, as_list=False)
                 if len(subset) == subset_size)
    if as_list:
        return list(generator)
    else:
        return generator

def asRelative(distribution):
    # get a list\dict of numbers (a distribution), return the relative distribution (element/sum)
    if 'values' in dir(distribution):
        # a dict type
        sm = float(sum(distribution.values()))
        return {k : v / sm for k,v in distribution.iteritems()}
    else:
        # a list type
        sm = float(sum(distribution))
        return [e/sm for e in distribution]

def allExcept(lst, index):
    # return all items in list except for list[index]
    return lst[:index] + lst[1 + index:]

def firstMismatch(list1, list2):
    return [(i, j) for i, j in zip(list1, list2) if i != j][0]

def weighted_mean(lst_of_item_weight):
    num = 0
    sum = 0.
    for item, weight in lst_of_item_weight:
        sum += item * weight
        num += weight
    return sum / num

def filterDict(filterFunc, orig_dict, byKey = True):
    """ Filter dict by key or by value.
    :param filterFunc: return True for items to keep, return false for items to filter out
    :param dict: original data 
    :param byKey: True to filter by keys, False to filter by value
    :return: a dict which is a subset of orig dict
    """
    tupleIndex = 0 if byKey else 1  # desired index in the (key value) tuple for filtering
    return dict(filter(lambda item: filterFunc(item[tupleIndex]), orig_dict.iteritems()))

def replaceKeys(orig_dict, oldKeys2NewKeys, inplace=True):
    """ replace keys with new keys using oldKeys2NewKeys mapping. """
    target_dict = orig_dict if inplace else {}
    for oldKey, newKey in oldKeys2NewKeys.items():
        target_dict[newKey] = orig_dict.get(oldKey)
        if inplace: orig_dict.pop(oldKey)
    return target_dict

def simpleReverseMap(origMap):
    return {v:k for k,v in origMap.iteritems()}

def reverseMap(origMap):

    # if values are unique, simpleReverseMap is sufficient
    if len(set(origMap.values())) == len(origMap):
        return simpleReverseMap(origMap)
    # otherwise, return a map to list of keys providing that value
    from collections import defaultdict
    out = defaultdict(list)
    for k,v in origMap.iteritems():
        out[v].append(k)
    return out

def listSplit(lst, delimeterElement):
    # as str.split(); return a splitted list (list of sub-lists), splitted by the delimeter
    return[list(y) for x, y in itertools.groupby(lst, lambda z: z == delimeterElement) if not x]

def randomSample(lst, n):
    import random
    lst_copy = lst[:]
    random.shuffle(lst_copy)
    return lst_copy[:n]

def nDisjointSubsetPairs(lst, n):
    # given a basic list, return n-sized list of pairs of two disjoint subset in the size of len/2
    lst=lst[:]
    set_size = len(lst)
    subset_size = set_size/2
    disjoint_pairs = []
    for i in range(n):
        first_subset = randomSample(lst, subset_size)
        complement_subset = [e for e in lst if e not in first_subset][:subset_size]
        disjoint_pairs.append((first_subset, complement_subset))
    return disjoint_pairs

def takeKeys(dic, keys):
    return {k: v for k, v in dic.iteritems() if k in keys}

def dictOfLists(pairs):
    # return a { key : [values given to that key] } for the pair list.
    # e.g. dictOfLists( [(0, "r"), (4, "s"), (0, "e")])  will return {0: ["r", "e"], 4: ["s"]}
    from collections import defaultdict
    r = defaultdict(list)
    for k,v in pairs:
        r[k].append(v)
    return dict(r)

# Evaluation and Statistics utils

def compute_F1PrR(predicted_count, gold_count, match_count):
    # return F1, precision and recall, based on overall counters and matching elements counter
    if not predicted_count and not gold_count:
        return 1,1,1
    elif not predicted_count and gold_count:
        return 0,1,0
    elif predicted_count and not gold_count:
        return 0,0,1
    precision = float(match_count) / predicted_count
    recall = float(match_count) / gold_count
    f1 = 0 if precision + recall == 0 \
        else (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def comp(graphs_dict, gold_dict, match_func=lambda g1, g2: g1.count_strict_match(g2)):
    """
    Compare two dicts of graphs {sentence-id:SentenceGraph} using comp_func. 
    Return the averaged F1, precision, recall.
    
    If gold-dict is a list (of gold_dicts) - return list of results for each gold_standard. 
    :param graphs_dict: "Predicted" graphs
    :param gold_dict: "Gold-Standard" graphs, or a list of those.
    :param match_func: edge-match counting function: 
        (predicted-SentenceGraph, gold-SentenceGraph) => (#self edges, #gold edges, #matched edges)
    :return: averaged (f1, precision, recall) over all sentences (Ids) both in graphs_dict and in gold_dict
    """
    import numpy as np
    if type(gold_dict) is dict:
        edge_counters = np.array([match_func(graphs_dict[i], gold_dict[i]) for i in graphs_dict if i in gold_dict])
        logging.debug("Evaluating for {} sentences...".format(len(edge_counters)))
        return compute_F1PrR(*tuple(edge_counters.sum(axis=0)))
    elif type(gold_dict) in [list, tuple]:
        return np.array([comp(graphs_dict, gold_dict_i, match_func=match_func)
                         for gold_dict_i in gold_dict])


# relaxed comparing graphs - predicted (e.g. crowd) to gold (e.g. expert, or SDP)
def laxcomp(graphs_dict, gold_dict):
    return comp(graphs_dict, gold_dict, lambda g1, g2: g1.count_relaxed_match(g2))

def pairwise_formalism_comparison(list_of_formalisms, pairwise_comp_f=laxcomp):
    """ 
    return 3 matrices of F1, precision, recall - of pairwise comparison of formalisms.
    :param list_of_formalisms: iterable of dicts, each is {sentence_id : SentenceGraph } of a certain formalism
    In the tables, the rows stand for the "predicted" graphs (i.e., they are the predictors),
    while the columns stand for the "gold-standard" graphs (i.e., they are being compared to, as gold standard)
    """
    f1_table, precision_table, recall_table = [], [], []
    for formalism_as_predicted in list_of_formalisms:
        f1_row, precision_row, recall_row = [], [], []
        for formalism_as_gold in list_of_formalisms:
            f1, precision, recall = pairwise_comp_f(formalism_as_predicted, formalism_as_gold)
            f1_row.append(f1)
            precision_row.append(precision)
            recall_row.append(recall)
        f1_table.append(f1_row)
        precision_table.append(precision_row)
        recall_table.append(recall_row)
    return np.array(f1_table), np.array(precision_table), np.array(recall_table)

def diameterToLengthCorrelation(graphs_dict):
    """
    :param graphs_dict: 
    :return: The Pearson correlation between sentence length and graph diameter 
    """
    length_diameter_pairs = [(len(g.tokSent), g.diameter()) for g in graphs_dict.values() if g.is_connected()]
    lengths, diameters = zip(*length_diameter_pairs)
    return np.corrcoef(lengths, diameters)[1,0]

def plotCounterAsPie(counter, relative=False):
    # plot counter as pie-chart
    import matplotlib.pyplot as plt

    # Data to plot
    if relative:
        counter = asRelative(counter)
    labels, values = zip(*counter.items())
    colors = ['r','b','g', 'c', 'gold', 'yellowgreen', 'lightcoral', 'lightskyblue','w']

    # Plot
    plt.pie(values, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()

def plotCounterAsBar(counter, relative=False):
    # plot Counter as bar-chart
    import numpy as np
    import matplotlib.pyplot as plt
    if relative:
        counter = asRelative(counter)
    labels, values = zip(*counter.items())
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()

def plotCounter(counter, chart="bar", relative=False):
    {"bar": plotCounterAsBar,
     "pie": plotCounterAsPie}[chart](counter, relative)

def plot_bars_2D(means2D, sd2D=None, axis1Labels=None, axis2Labels=None, yAxisLabel="Score", xAxisLabel="Group", title=""):
    import numpy as np
    import matplotlib.pyplot as plt

    means2D = np.array(means2D)
    sd2D = np.array(sd2D) if sd2D else None
    n_groups = means2D.shape[1]

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors = ['r','b','g', 'c', 'y','k','w']

    rects={}
    if not axis1Labels:
        axis1Labels=[unicode(i) for i in range(means2D.shape[0])]
    for i,ax1value in enumerate(axis1Labels):
        rects[ax1value] = plt.bar(index  +(i*bar_width), means2D[i], bar_width,
                         alpha=opacity,
                         color=colors[i],
                         yerr=sd2D[i] if sd2D else None,
                         error_kw=error_config,
                         label=ax1value)

    plt.xlabel(xAxisLabel)
    plt.ylabel(yAxisLabel)
    plt.title(title)
    if axis2Labels:
        plt.xticks(index + bar_width / 2, axis2Labels)
    plt.legend()

    plt.tight_layout()
    plt.show()

# pretty print table in tabular format
def prettyPrintTable(table, rowLabels=None, colLabels=None, afterDot= None, justify = "R", columnWidth = 0):
    # Not enforced but
    # if provided columnWidth must be greater than max column width in table!
    if columnWidth == 0:
        # find max column width
        for row in table:
            for col in row:
                width = len(str(col))
                if width > columnWidth:
                    columnWidth = width
    # we're going to append strings, so need primitive list (not numpy array)
    import numpy as np
    if type(table) is np.ndarray:
        table = table.tolist()
    # this is how to represent a single row (list of items)
    def reprRow(row):
        rowList = []
        for col in row:
            if afterDot and type(col) is float:
                col = round(col, afterDot)
            if justify == "R": # justify right
                rowList.append(str(col).rjust(columnWidth))
            elif justify == "L": # justify left
                rowList.append(str(col).ljust(columnWidth))
            elif justify == "C": # justify center
                rowList.append(str(col).center(columnWidth))
        return ' '.join(rowList)
    # now build full table repr
    outputStr = ""
    if colLabels:
        if rowLabels:
            outputStr += reprRow(["--"] + colLabels) + "\n"
        else:
            outputStr += reprRow(colLabels) + "\n"
    for i,row in enumerate(table):
        if rowLabels:
            row = [rowLabels[i]] + row
        outputStr += reprRow(row) + "\n"
    return outputStr

def tabularToCsv(tabular_data, output_csv_file=None):
    """ takes any tabular data format -- nested lists, tuples, np.array, etc - and output CSV format.
    Outmost level is rows, rest are flatten.
    :param output_csv_file: if specified, CSV is saved there.
    :returns the string representation of the data (CSV)
    """
    s = ''
    for row in tabular_data:
        row_repr = ', '.join(str(e) for e in flatten(row, recursively=True))
        s += row_repr + '\n'
    if output_csv_file:
        with open(output_csv_file, "w") as f:
            f.write(s)
    return s


def visualize_graphs(graphs_dict, destination_path="vis/currentGraphs", postfix="crowd", **kwargs):
    """ Create visalizations for all graphs in graph_dict
    :param graphs_dict: {sentence_id : SentenceGraph }
    :param destination_path: where to place the created html visualization files
    :return: Nothing
    """
    import os.path
    for id, graph in graphs_dict.items():
        vis_fn = str(id) + "_" + postfix + ".html"
        vis_full_path = os.path.join(destination_path, vis_fn)
        graph.visualize(vis_full_path, **kwargs)


class Span:
    def __init__(self, (beginIndex, endIndex)):
        # assuming inclusive spans (as in annotated data) - e.g. (0,1) includes both word 0 and word 1
        self.beginIndex = beginIndex
        self.endIndex = endIndex    # included in span


    def toSet(self):
        return set(range(self.beginIndex, self.endIndex+1)) # assuming inclusive spans (as in annotated data)

    def is_within(self, span2):
        # is span1 strictly within span
        return self.beginIndex >= span2[0] and self.endIndex <= span2[1]

    def __len__(self):
        # length of span
        return self.endIndex - self.beginIndex + 1 # +1 because endIndex is inclusive

    def __repr__(self):
        return "Span: [{}, {})".format(self.beginIndex, self.endIndex)

    @staticmethod
    def all_pairs(lst):
        for index, item in enumerate(lst):
            for item2 in lst[index+1:]:
                yield (item, item2)

    @staticmethod
    def order(span1, span2):
        # return the span that begin before the other (it's begin index is lower) as first, and the other as second
        return (span1, span2) if span1[0] <= span2[0] \
            else (span2, span1)

    @staticmethod
    def validSpan(span):
        return span[0] <= span[1]

    @staticmethod
    def intersect(span1, span2):
        # return the maximal overlapping span between the spans
        intersectingInterval = (max(span1[0],span2[0]), min(span1[1],span2[1]))
        return intersectingInterval if Span.validSpan(intersectingInterval) else None

    @staticmethod
    def word_to_char(tokSent, (word_start, word_end)):
        """
        For transforming word-indices span to char-indices span. 
        Given a tuple node indicating word start and end indices (exclusive end index!),
        return its corresponding char indices.
        tokSent is the tokenized sentence (list of words), corresponding to the word-indices.
        """
        word_end = word_end -1  # transpose it to inclusive end index
        return (word_start + sum(map(len, tokSent[: word_start])),
                word_end + sum(map(len, tokSent[: word_end])) + len(tokSent[word_end]))

class XSpan(Span):
    """ Exclusive Span class (as in SentenceGraph nodes), e.g. (0,1) includes only 0 but not 1. """
    def __init__(self, (beginIndex, endIndex)):
        # assuming exclusive indices.
        Span.__init__(self, (beginIndex, endIndex))

    def __repr__(self):
        return "XSpan: [{}, {}]".format(self.beginIndex, self.endIndex)

    def toSet(self):
        return set(range(self.beginIndex, self.endIndex)) # assuming exclusive spans (as in annotated data)

    def __len__(self):
        # length of span
        return self.endIndex - self.beginIndex  #  because endIndex is exclusive

# specifically for computing cost of QASemDep Turking project
def compute_configuration_cost_per_sentence(num_of_gen, num_of_val):
    #in cents
    genCostPerSentence = 122
    valCostPerSentence = 80     # if no more than 2 generators aggregated
    addValCostPerGeneratorAggregatedAbove2 = 35
    if num_of_gen<=2:
        return genCostPerSentence*num_of_gen + valCostPerSentence*num_of_val
    else:
        valCostPerSentence = valCostPerSentence + addValCostPerGeneratorAggregatedAbove2*(num_of_gen-2)
        return genCostPerSentence*num_of_gen + valCostPerSentence*num_of_val


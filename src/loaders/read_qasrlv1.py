"""
usage: read_qasrlv1.py --in=QA-SRL.1.0-input-file 

Handle QASRL.1.0 (He et. al., 2015) data format (CoNLL like). Contains functions for: 
* load QASRL data format (v1) from file
* put into a self-defined JSON data-structure (see below)
* put into SentenceGraphs 
* visualize using Brat

JSON data-structures (self-defined):

the json data-structure for a QasrlSentence:
 {  "sentence" : <the sentence>,
    "predicates" : [ list of QasrlPredicates ] }

the json data-structure for a QasrlPredicate:
 {  "sentence" : <the sentence>,
    "predicate": <the predicate>,
    "idx":  (tuple of the predicate indices),
    "QAs": [list of QasrlQAs ] }

the json data-structure for a QasrlQA:
{   "Q": <the question string>,
    "A": [ a non-empty list of QasrlAnswer ] }
    
the json data-structure for a QasrlAnswer:
{   "span": <answer string>,
    "idx": (binary tuple of the answer-span indices) }
"""

import logging, json

def load_qa_srl_v1_data(filename):
    """ return a list of QasrlSentence (json object\python dict) """
    QasrlSentences = []
    with open(filename, "r") as f:
        block = ""
        for line in f:
            if not line.strip(): # new block
                QasrlSentence = read_QasrlSentence(block)
                QasrlSentences.append(QasrlSentence)
                logging.debug("collected annotation for sentence: " + QasrlSentence["sentence"])
                block = ""

            else:
                block += line
    return QasrlSentences

def read_QasrlSentence(block):
    """
    the json data-structure for a QasrlPredicate:
     {  "sentence" : <the sentence>,
        "predicate": <the predicate>,
        "idx":  (tuple of the predicate indices),
        "QAs": [list of QasrlQAs ] }
    """
    QasrlPredicates = []
    QasrlPredicate = None
    block_lines = [line.strip() for line in block.splitlines() ]
    sentence = block_lines[1]
    num_of_sentences_left_for_predicate = 0 # a countdown counter for each predicate
    for line in block_lines[2:]:  # skip first line (PROPBANK marker) and second line (sentence string)
        if num_of_sentences_left_for_predicate==0:  # then this line is declaring a new predicate
            # append last QasrlPredicate and start a new one
            QasrlPredicates.append(QasrlPredicate)
            # extract info from declaring line
            assert len(line.split()) == 3, "predicate-declaring line expected, more than 3 token found"
            predicate_index, predicate, num_of_sentences_left_for_predicate = line.split()
            predicate_index = int(predicate_index)
            num_of_sentences_left_for_predicate = int(num_of_sentences_left_for_predicate)
            # create new QasrlPredicate
            QasrlPredicate = {"sentence":sentence,
                              "predicate": predicate,
                              "idx": word_index_to_char_span(sentence, predicate_index),
                              "QAs": []}
        else:
            QasrlPredicate["QAs"].append(read_QasrlQA(line, sentence))
            num_of_sentences_left_for_predicate -= 1
    return {"sentence": sentence,
            "predicates": QasrlPredicates[1:]}  # the first is always None

def word_index_to_char_span(full_string, word_index):
    tokens = dict(enumerate(full_string.split()))
    wi2st = {}  # word index to (char-index) span tuple
    char_index_counter = 0
    for wi, tok in tokens.iteritems():
        wi2st[wi] = (char_index_counter, char_index_counter + len(tok))
        char_index_counter += len(tok) + 1
    return wi2st[word_index]


def read_QasrlQA(line, sentence):
    """
    the json data-structure for a QasrlQA:
    {   "Q": <the question string>,
        "A": [ a non-empty list of QasrlAnswer ] }
    """
    tokens = line.split("\t")
    raw_answer = tokens[-1]
    question = ' '.join([word for word in tokens[:-1] if word != "_"])
    QasrlAnswers = read_QasrlAnswers(raw_answer, sentence)
    return {"Q": question,
            "A": QasrlAnswers}


def read_QasrlAnswers(raw_answer_string, sentence):
    """
    the json data-structure for a QasrlAnswer:
    {   "span": <answer string>,
        "idx": (binary tuple of the answer-span indices) }
    """
    QasrlAnswers = []
    for answer in raw_answer_string.split(" ### "):
        if answer in sentence:
            start_i = sentence.index(answer)
            end_i = start_i + len(answer)
            idx = (start_i, end_i)
        else:
            idx = None
        QasrlAnswers.append({
            "span": answer,
            "idx": idx
        })
    return QasrlAnswers


def qa_srl_file_to_json(qa_srl_fn, target_json_fn):
    """ transfer QA-SRL data file to a json file. """
    QasrlSentences = load_qa_srl_v1_data(qa_srl_fn)
    json.dump(QasrlSentences, target_json_fn)


def qasrlSentence_to_SentenceGraph(qasrlSentence):
    from SentenceGraph import SentenceGraph
    g = SentenceGraph(qasrlSentence["sentence"], word_indexed=False)
    for qasrlPredicate in qasrlSentence["predicates"]:
        predicate_indices = qasrlPredicate["idx"]
        for QasrlQA in qasrlPredicate["QAs"]:
            for QasrlAnswer in QasrlQA["A"]:
                g.add_edge(predicate_indices, QasrlAnswer["idx"], label=QasrlQA["Q"])
    return g


def qasrlPredicate_to_SentenceGraph(qasrlPredicate):
    from SentenceGraph import SentenceGraph
    g = SentenceGraph(qasrlPredicate["sentence"], word_indexed=False)
    predicate_indices = qasrlPredicate["idx"]
    for QasrlQA in qasrlPredicate["QAs"]:
        for QasrlAnswer in QasrlQA["A"]:
            g.add_edge(predicate_indices, QasrlAnswer["idx"], label=QasrlQA["Q"])
    return g


def visualize_QasrlPredicate(QasrlPredicate, out_fn):
    """ Create html visualization of an QA-SRL annotation of a single QasrlPredicate. """
    from SentenceGraph import SentenceGraph
    g = SentenceGraph(QasrlPredicate["sentence"], word_indexed=False)
    predicate_indices = QasrlPredicate["idx"]
    for QasrlQA in QasrlPredicate["QAs"]:
        for QasrlAnswer in QasrlQA["A"]:
            g.add_edge(predicate_indices, QasrlAnswer["idx"], label=QasrlQA["Q"])
    # create visualization file
    g.visualize(out_fn)
    return g    # return SentenceGraph with annotations of only a single predicate


def read_qasrlv1_data_to_SentenceGraphs(qasrlv1CoNLLFileName):
    QasrlSentences = load_qa_srl_v1_data(qasrlv1CoNLLFileName)
    return [qasrlSentence_to_SentenceGraph(qasrlSentence)
            for qasrlSentence in QasrlSentences]


# example usage
if __name__ == "__main__":
    # Parse arguments
    from utils import *
    from docopt import docopt
    args = docopt(__doc__)
    inp_fn = args["--in"]
    """
    Reminder:
    the json data-structure for a QasrlSentence:
     {  "sentence" : <the sentence>,
        "predicates" : [ list of QasrlPredicates ] }
    QasrlSentences is a list of QasrlSentence objects.
    """
    QasrlSentences = load_qa_srl_v1_data(inp_fn)
    data=QasrlSentences
    allQAs = [QA for S in QasrlSentences for P in S["predicates"] for QA in P["QAs"] ]
    allAnswers = [a["span"] for qa in allQAs for a in qa["A"]]

    """
    Example usage for data-loading and visualization:
    
    * using JSON structures:
    >>> p=QasrlSentences[0]["predicates"][0]
    >>> visualize_QasrlPredicate(p, "pic.html")
    
    * using SentenceGraphs:
    >>> sentenceGraphs = read_qasrlv1_data_to_SentenceGraphs(inp_fn)
    >>> g = sentenceGraphs[0]
    >>> g.visualize("qasrlv2.html")
    """
"""
usage: read_qasrlv2.py --in=QA-SRL.2.0-input-file.jsonl 

Handle QASRL.2.0 (2018) data (JSON-lines). Contains functions for: 
* load QASRL.2.0 data - each line is a JSON defined here: https://github.com/uwnlp/qasrl-bank/README.md
* put into SentenceGraphs 
* visualize using Brat
"""

import json_lines, sys
from collections import Counter


def read_qasrlv2_data_to_SentenceGraphs(qasrlv2_jsonl_fn):
    """ Return the QA-SRL V2 data as a list of SentenceGraphs. """
    return [qasrlv2_sentence_to_SentenceGraph(qasrlSent)
            for qasrlSent in read_jsonl_file(qasrlv2_jsonl_fn)]

def read_jsonl_file(jsonl_fn):
    with open(jsonl_fn, 'rb') as f:
        for item in json_lines.reader(f):
            yield item


def countValid(genQuestionData):
    """ 
    return the number of validators that judged the question as valid. 
    :param genQuestionData: a value from the 'questionLabels' dict inside a verbEntry
    """
    return len([val for val in genQuestionData['answerJudgments'] if val['isValid']])

def countPercValid(genQuestionData):
    """ 
    :return the percentage of validators that judged the question as valid. 
    :param genQuestionData: a value from the 'questionLabels' dict inside a verbEntry
    """
    return float(countValid(genQuestionData)) / len([val for val in genQuestionData['answerJudgments']])

def getGoodAnswerSpans(genQuestionData, agreementThreshold=0.66):
    """
    :return list of answer spans that have sufficient agreement. 
    :param genQuestionData: a value from the 'questionLabels' dict inside a verbEntry
    :param agreementThreshold: The threshold of agreement rate above which spans are returned.
     Agreement rate is computed relative to overall number of validators (including those who judged the question as invalid)
    """
    spans = [tuple(span) for val in genQuestionData['answerJudgments'] if val['isValid'] for span in val['spans']]
    span2count = dict(Counter(spans))   # span to number of validators that give the span
    span2agreement = {s : float(c) / len(genQuestionData['answerJudgments']) for s,c in span2count.iteritems()}
    return [span for span, agreement in span2agreement.iteritems() if agreement>agreementThreshold]


def qasrlv2_sentence_to_SentenceGraph(sentenceQasrlV2):
    """
    Insert the QASRL data into networkx graph wrapper (SentenceGraph).
     Edges are between verbs and well-agreed answer spans
    :param sentenceQasrlV2: a JSON of QASRLV2 of a single sentence
    """
    from SentenceGraph import SentenceGraph
    sentenceTok = sentenceQasrlV2["sentenceTokens"]
    g = SentenceGraph(sentenceTok, word_indexed=True)
    for verbIndex, verbEntry in sentenceQasrlV2["verbEntries"].iteritems():
        verbIndex = (int(verbIndex), int(verbIndex)+1)
        for questionString, question in verbEntry['questionLabels'].iteritems():
            goodSpans = getGoodAnswerSpans(question)  # get agreed answer spans
            for answerSpan in goodSpans:
                g.add_edge(verbIndex, answerSpan, questionString)
    return g

# helper function
def visualize_qasrlv2_sentence(sentenceQasrlV2, out_fn):
    sentenceGraph = qasrlv2_sentence_to_SentenceGraph(sentenceQasrlV2)
    # visualize
    sentenceGraph.visualize(out_fn)


if __name__ == "__main__":
    from utils import *
    from docopt import docopt
    args = docopt(__doc__)
    input_fn = args["--in"]
    # or: input_fn = "QASRL.2.0/dev.jsonl"
    data = list(read_jsonl_file(input_fn))
    # data is list of sentenceQasrlV2, a JSON defined here: https://github.com/uwnlp/qasrl-bank/README.md
    # enrich data to have also "sentence" key
    for d in data:
        d["sentence"]=' '.join(d["sentenceTokens"])
    g = qasrlv2_sentence_to_SentenceGraph(data[0])







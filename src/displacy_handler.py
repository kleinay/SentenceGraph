
import sys, logging
logging.basicConfig(level = logging.DEBUG)
# meantime there is a bug on spaCy.displacy regarding arrow direction, so we'll load our local spacy where it's fixed
# after spaCy will fix it, we can load regularly (delete the next 2 lines)
# sys.path=["/home/ir/kleinay/spaCy"]+sys.path
from spacy import displacy


def output_displacy_svg(tokSent, nx_digraph, out_fn, edgeLabelProcessFunc=lambda s:s):
    displacy_dep_input = get_displacy_dep_input(tokSent, nx_digraph, edgeLabelProcessFunc)

    svg = displacy.render(displacy_dep_input, style='dep', manual='True')
    # write svg text to a file
    with open(out_fn, "w") as f:
        f.write(svg)
    logging.debug("output written to: {}".format(out_fn))


def get_displacy_dep_input(tokSent, nx_digraph, edgeLabelProcessFunc=lambda s:s):
    def get_arc(u, v, data):
        direction = u'right' if u[0] < v[0] else u'left'
        return {'dir': direction,
                'start': min(u[0],v[0]),
                'end': max(u[0],v[0]),
                'label': edgeLabelProcessFunc(data['label'])}

    words = [ {"text" : w, "tag":""} for w in tokSent]
    arcs = [get_arc(*edge) for edge in nx_digraph.edges.data()]
    return {'words':words,
                          'arcs':arcs}

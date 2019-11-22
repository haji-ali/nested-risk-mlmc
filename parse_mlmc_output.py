# Based on notes from
# https://www.vipinajayakumar.com/parsing-text-with-python/#parsing-text-in-complex-format-using-regular-expressions

import re
import numpy as np

reg_num = r"[+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?"
rx_dict = {
    "Comment" : re.compile(r"^\s*#(?P<comment>.*)$"),
    "Linesep" : re.compile(r"^\+(?:-*\+)*\s*$"),
    "Header" : re.compile(r"^\|(?:\s*(?:[A-Za-z]*)\s*\|)*\s*$"),
    "KeyValue" : re.compile(r"^\s*(?P<key>[^=:]*)\s*(?:\:|=)\s*(?P<val>{num})\s*(?P<rem>.+)?\s*$".format(num=reg_num)),
    "Level" : re.compile(r"^\|\s*(?P<inactive>\*)?\s*(?P<level>\d*)\s*\|(?P<data>(?:\s*{num}\s*\|)*)\s*$".format(num=reg_num)),
}

class mlmc_iter:
    def __init__(self):
        self.headers = dict()
        self.keyval = dict()
        self.data = None
        self.starting_lvl = 0

    def set_headers(self, headers):
        assert(self.data is None)
        for i, h in enumerate(headers):
            self.headers[h] = i
        self.data = np.zeros((0, len(headers)))

    def add_lvl(self, lvl):
        assert(self.data.shape[0] == lvl)
        self.data.resize((lvl+1, self.data.shape[1]), refcheck=False)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, key):
        if key in self.headers:
            return self.data[:, self.headers[key]]
        else:
            return self.keyval[key]

    def __contains__(self, key):
        return key in self.headers or key in self.keyval

def split(s, sp):
     return [p.strip() for p in s.split(sp) if len(p.strip()) > 0]

def parse_line(line):
    for key, rx in rx_dict.items():
        match = rx.match(line)
        if match:
           return key, match
    raise Exception("Unrecognized line: " + line)

def parse_lines(lines):
    itrs = []
    itr = None
    for l in lines:
        l = re.sub("\033\\[\d+(;\d+)*m", "", l)   # Remove any formatting tags
        key, match = parse_line(l)
        if key in ("Comment", "Linesep"):
            continue
        elif key == "Header":
            itr.set_headers(split(l, "|")[1:])  # First header is lvl
        elif key == "KeyValue":
            if match.group("key") == "TOL":   # First field on a new iteration
                itr = mlmc_iter()
                itrs.append(itr)
            itr.keyval[match.group("key")] = float(match.group("val"))
        elif key == "Level":
            lvl = int(match.group("level"))
            itr.add_lvl(lvl)
            if match.group("inactive") is not None:
                itr.starting_lvl = lvl+i
            itr.data[lvl, :] = [float(x) for x in split(match.group("data"), "|")]
    return itrs

def parse_file(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
    return parse_lines(lines)


if __name__ == "__main__":
    from mimclib import ipdb
    ipdb.set_excepthook()

    itrs = parse_file("build/test.txt")

    ipdb.embed()

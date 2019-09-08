import numpy as np


class Dependency:
    def __init__(self, id, form, lang, tag, head, rel, tgt_id=-1):
        self.id = id
        self.org_form = form.lower()
        self.form = form.lower() + "_" + lang
        self.lang = lang
        self.tag = tag
        self.head = head
        self.rel = rel
        self.tgt_id = tgt_id

    def __str__(self):
        values = [str(self.id), self.org_form, self.lang, self.tag, "_", "_", str(self.head), self.rel, "_", "_"]
        return '\t'.join(values)


def read_deptree(infile):
    lines = []
    while True:
        line = infile.readline().strip()
        if line is None or line == '':
            break
        lines.append(line)
    sent_len = len(lines) + 1
    if sent_len == 1:
        return None

    sentence = [Dependency(0, "--choose--", "_", "_", 0, "_")]
    for line in lines:
        tok = line.strip().split('\t')
        if len(tok) != 10:
            raise Exception('error: ' + line)
        sentence.append(Dependency(int(tok[0]), tok[1], tok[2], tok[3], int(tok[6]), tok[7], -int(tok[0])))

    return sentence


def read_parallel_corpus(infile_parallel, infile_align):
    line = infile_parallel.readline().strip()
    if line is None or line == '':
        raise Exception('error: ' + line)
    bi_sent = line.split('|||')
    unit_len = len(bi_sent)
    if unit_len != 2:
        raise Exception('error: ' + line)

    src_words = bi_sent[0].strip().split(' ')
    tgt_words = bi_sent[1].strip().split(' ')
    src_len, tgt_len = len(src_words), len(tgt_words)
    align_probs = read_align(infile_align, src_len, tgt_len)

    return src_words, tgt_words, align_probs


def read_align(infile_align, src_len, tgt_len):
    align_probs = np.full((tgt_len, src_len + 1), 0.0, dtype=np.float64)
    index = int(0)
    while index < tgt_len:
        line = infile_align.readline().strip()
        if line is None or line == '':
            raise Exception('tgt length not match: curr %d, total %d' % (index, tgt_len))
        content_pieces = line.split('\t')
        content_index = int(content_pieces[0])
        if content_index != index:
            raise Exception('align file error: content_index %d, actual_index %d' % (content_index, index))
        for i in range(1, len(content_pieces)-1):
            atom_values = content_pieces[i].split(' ')
            if len(atom_values) != 2:
                raise Exception('invalid source index and prob: '  + content_pieces[i])
            src_index = int(atom_values[0])
            src_prob = float(atom_values[1])
            if src_index == -1:
                src_index = src_len
            align_probs[index, src_index] = src_prob
        index = index + 1

    line = infile_align.readline().strip()
    if line is not None and line != '':
        raise Exception('last line is not empty:' + line)

    for index in range(1, tgt_len):
        align_probs[index, 0] = -1.0

    return align_probs


def read_next_instance(depfile, parallel_file, align_file):
    deptree = read_deptree(depfile)
    if deptree is None:
        return None

    src_words, tgt_words, align_probs = read_parallel_corpus(parallel_file, align_file)

    return deptree, src_words, tgt_words, align_probs


def write_deptree(outfile, sentence):
    for entry in sentence:
        if entry.id > 0: outfile.write(str(entry) + '\n')
    outfile.write('\n')


def reorder_deptree(sentence, new_seqs):
    length = len(sentence)
    newids = np.full((length), -1, dtype=np.int)
    for idx, new_idx in zip(range(length), new_seqs):
        newids[new_idx] = idx
    newsent = []
    for idx, new_idx in zip(range(length), new_seqs):
        dep = sentence[new_idx]
        curhead = newids[dep.head]
        newdep = Dependency(idx, dep.org_form, dep.lang, dep.tag, \
                                  curhead, dep.rel, dep.tgt_id)
        newsent.append(newdep)

    return newsent


def del_node_deptree(sentence, del_idx):
    length = len(sentence)
    newids = np.full((length), -1, dtype=np.int)
    for index in range(del_idx):
        newids[index] = index
    newids[del_idx] = -1
    for index in range(del_idx + 1, length):
        newids[index] = index - 1

    del_idx_head = sentence[del_idx].head
    newsent = []
    for index in range(del_idx):
        dep = sentence[index]
        curhead = dep.head
        if curhead == del_idx: curhead = del_idx_head
        rhead = newids[curhead]
        newdep = Dependency(dep.id, dep.org_form, dep.lang, dep.tag, \
                            rhead, dep.rel, dep.tgt_id)
        newsent.append(newdep)

    for index in range(del_idx+1, length):
        dep = sentence[index]
        curhead = dep.head
        if curhead == del_idx: curhead = del_idx_head
        rhead = newids[curhead]
        newdep = Dependency(dep.id-1, dep.org_form, dep.lang, dep.tag, \
                            rhead, dep.rel, dep.tgt_id)
        newsent.append(newdep)

    return newsent


def add_tform_fake_deptree(sentence, tform, tlang, src_id, tgt_id):
    newsent = []
    sent_len = len(sentence)
    index = 0
    while sentence[index].id < src_id:
        dep = sentence[index]
        newsent.append(dep)
        index += 1

    dep = sentence[index]
    tag, head, rel = dep.tag, dep.head, dep.rel

    if dep.lang == tlang:
        while dep.id == src_id and dep.lang == tlang:
            newsent.append(dep)
            index += 1
            if index < sent_len: dep = sentence[index]
            else: break
        newdep = Dependency(src_id, tform, tlang, tag, head, rel, tgt_id)
        newsent.append(newdep)
    else:
        newdep = Dependency(src_id, tform, tlang, tag, head, rel, tgt_id)
        newsent.append(newdep)
        index += 1

    while index < sent_len:
        newsent.append(sentence[index])
        index += 1

    return newsent


def normalize_fake_deptree(sentence):
    length = len(sentence)
    last_idx = sentence[length-1].id
    newheads = np.full((last_idx+1), -1, dtype=np.int)
    for index in range(length):
        cur_id = sentence[index].id
        next_id = -1
        if index < length - 1: next_id = sentence[index+1].id
        if cur_id != next_id: newheads[cur_id] = index
        else: sentence[index].head = -1

    newsent = [sentence[0]]
    for index in range(1, length):
        dep = sentence[index]
        curhead = index + 1
        if dep.head != -1: curhead = newheads[dep.head]
        newsent.append(Dependency(index, dep.org_form, dep.lang, dep.tag, \
                                  curhead, dep.rel, dep.tgt_id))

    return newsent


def generate_target_orders(sentence):
    new_order = []
    length = len(sentence)
    index = 0
    while index < length:
        if sentence[index].tgt_id < 0:
            new_order.append(index)
            index += 1
        else:
            span = [(index, sentence[index].tgt_id)]
            idx = 1
            while index+idx < length and sentence[index+idx].tgt_id >= 0:
                span.append((index+idx, sentence[index+idx].tgt_id))
                idx += 1
            span.sort(key=lambda tup: tup[1])
            for (seq_idx, tgt_id) in span:
                new_order.append(seq_idx)
            index = index + idx

    return new_order



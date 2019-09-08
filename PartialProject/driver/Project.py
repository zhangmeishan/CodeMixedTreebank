import sys
sys.path.extend(["../","./"])
import time
import argparse
from driver.Corpus import *
from collections import Counter
import copy


def evaluate(dep_file, parallel_file, align_file, outputFile, percentage, tlang):
    start = time.time()
    infile_dep = open(dep_file, 'r', encoding='UTF-8')
    infile_parallel = open(parallel_file, 'r', encoding='UTF-8')
    infile_align = open(align_file, 'r', encoding='UTF-8')
    output = open(outputFile, 'w', encoding='utf-8')

    processed_count, valid_count = 0, 0
    del_src_word_counter = Counter()
    del_tgt_word_counter = Counter()
    temp_tgt_count = 0
    while True:
        next_data = read_next_instance(infile_dep, infile_parallel, infile_align)
        if next_data is None:
            break
        deptree, src_words, tgt_words, align_probs = next_data
        processed_count += 1
        src_len, tgt_len = len(src_words), len(tgt_words)
        if len(deptree) != src_len: continue
        valid = True
        for entry, src_word in zip(deptree, src_words):
            if entry.org_form != src_word:
                valid = False
                break
        if valid is False: continue

        projects, sorted_projects, invalids, src_del_probs = \
            sort_tgt2src_prob(align_probs, src_words, tgt_words)
        for idx in range(1, tgt_len):
            if projects[idx] == 0:
                valid = False
                break
        if valid is False: continue

        for idx in invalids:
            del_tgt_word_counter[tgt_words[idx]] += 1
            temp_tgt_count += 1

        # Word Substitution
        valid_len = len(sorted_projects)
        tgt_sub_len = np.min([valid_len, int(valid_len * percentage + 0.1)])
        selected_ids = set()
        for idx in range(tgt_sub_len):
            selected_ids.add(sorted_projects[idx][0])

        newtree = copy.deepcopy(deptree)
        for idx in range(1, tgt_len):
            if idx in selected_ids:
                newtree = add_tform_fake_deptree(newtree, tgt_words[idx], tlang, projects[idx], idx)

        valid_deptree = normalize_fake_deptree(newtree)

        have_root = False
        for dep in valid_deptree:
            if dep.id > 0 and dep.head == 0:
                have_root = True
                break
        if have_root is False:
            print("check reason")

        # Word Deletion
        null_src_len = len(src_del_probs)
        del_src_len = np.min([null_src_len, int(null_src_len * percentage + 0.1)])
        for idx in range(del_src_len):
            (src_idx, reserve_prob) = src_del_probs[idx]
            del_idx = -1
            for curdep in valid_deptree:
                if curdep.id > 0 and curdep.tgt_id == -src_idx:
                    del_idx = curdep.id
                    break
            if del_idx == -1:
                print("strange")
                break
            del_src_word_counter[valid_deptree[del_idx].org_form] += 1
            valid_deptree = del_node_deptree(valid_deptree, del_idx)

        # Sentence Reordering
        new_order = generate_target_orders(valid_deptree)
        valid_deptree = reorder_deptree(valid_deptree, new_order)

        write_deptree(output, valid_deptree)
        valid_count += 1

        if valid_count % 1000 == 0:
            current = time.time()
            eclipsed_time = float(current - start)
            print("%d sentences processed, %d valid, eclipsed time = %.2f" \
                  %(processed_count, valid_count, eclipsed_time))

    infile_dep.close()
    infile_parallel.close()
    infile_align.close()
    output.close()

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d, %d valid, parser time = %.2f" % (processed_count, valid_count, during_time))

    # print("src_tgt_word_counter: ")
    total_del_src = 0
    for delword, count in del_src_word_counter.most_common():
        # print("%s\t%d" % (delword, count))
        total_del_src += count

    # print("del_tgt_word_counter: ")
    total_del_tgt = 0
    for delword, count in del_tgt_word_counter.most_common():
        # print("%s\t%d" % (delword, count))
        total_del_tgt += count

    print("Total del src words: %d, Total del tgt words: %d " % (total_del_src, total_del_tgt))


def sort_tgt2src_prob(align_probs, src_words, tgt_words):
    src_len, tgt_len = len(src_words), len(tgt_words)
    parse_preds = np.argmax(align_probs, axis=1)
    results = []
    invalids = []
    for tgt_id, src_id in zip(range(tgt_len), parse_preds):
        if tgt_id == 0: continue
        if src_id < src_len:
            if align_probs[tgt_id, src_len] < 1.2:
                results.append((tgt_id, src_id, align_probs[tgt_id, src_id]))
            elif src_words[src_id] == tgt_words[tgt_id]:
                results.append((tgt_id, src_id, align_probs[tgt_id, src_id]))
            else:
                invalids.append(tgt_id)
        else:
            invalids.append(tgt_id)

    results.sort(key=lambda tup: tup[2], reverse=True)

    src_probs = np.full((src_len), 0.0, dtype=np.float64)
    for idx in range(tgt_len):
        for idy in range(src_len):
            src_probs[idy] += align_probs[idx, idy]

    valid_srcs = set([0])
    for idx in range(1, tgt_len):
        src_idx = parse_preds[idx]
        if src_idx < src_len:
            valid_srcs.add(src_idx)

    src_results = []
    for idy in range(src_len):
        if idy not in valid_srcs:
            src_results.append((idy, src_probs[idy]))

    src_results.sort(key=lambda tup: tup[1])

    return parse_preds, results, invalids, src_results


if __name__ == '__main__':
    np.random.seed(666)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dep_file', default='en-ud-train.conll')
    parser.add_argument('--parallel_file', default='en-de-train.txt')
    parser.add_argument('--align_file', default='train.en-de.prob')
    parser.add_argument('--output_file', default='en-de.base.conll')
    parser.add_argument('--lang', default='pt')
    parser.add_argument('--percentage', default=1.2, type=float, help='percentage of words to be projected')

    args, extra_args = parser.parse_known_args()

    evaluate(args.dep_file, args.parallel_file, args.align_file, args.output_file, \
             args.percentage, args.lang)







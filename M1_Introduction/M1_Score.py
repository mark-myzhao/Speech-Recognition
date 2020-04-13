import argparse
import wer
import numpy as np

# create a function that calls wer.string_edit_distance() on every utterance
# and accumulates the errors for the corpus. Then, report the word error rate (WER)
# and the sentence error rate (SER). The WER should include the the total errors as well as the
# separately reporting the percentage of insertions, deletions and substitutions.
# The function signature is
# num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(ref=reference_string, hyp=hypothesis_string)
#
def score(ref_trn=None, hyp_trn=None):
    separator = '('  # for the TRN format

    # build the lookup table for the reference
    ref_tb = dict()
    with open(ref_trn) as ref:
        for ref_line in ref:
            value_key = ref_line.strip().split(separator)
            if len(value_key) == 2:
                v, k = value_key[0].strip().split(), value_key[1][:-1]
                ref_tb[k] = v
            else:
                raise RuntimeError(f'TRN file format required for: {ref_trn}')

    word_rec = np.zeros(5, dtype=np.int64)  # total_words, err_words, del_words, ins_words, sub_words
    sentence_rec = np.zeros(2, dtype=np.int64)  # total_sentences, err_sentences
    with open(hyp_trn) as hyp:
        for hyp_line in hyp:
            value_key = hyp_line.strip().split(separator)
            if len(value_key) == 2:
                v, k = value_key[0].strip().split(), value_key[1][:-1]
                ref_v = ref_tb.get(k, '')
                num_tokens, num_err, num_del, num_ins, num_sub = wer.string_edit_distance(ref=ref_v, hyp=v)
                if num_tokens > 0 and num_err > 0:
                    sentence_rec += [1, 1]
                    word_rec += [num_tokens, num_err, num_del, num_ins, num_sub]
                else:
                    sentence_rec += [1, 0]
                    word_rec += [num_tokens, 0, 0, 0, 0]
            else:
                raise RuntimeError(f'TRN file format required for: {hyp_trn}')

    print('-----------------------------------')
    print('Sentence Error Rate:')
    print(f'Sum: N={sentence_rec[0]}, Err={sentence_rec[1]}')
    print(f'Avg: N={sentence_rec[0]}, Err={(sentence_rec[1] / sentence_rec[0]) * 100:.2f}%')

    print('-----------------------------------')
    print('Word Error Rate:')
    print('Sum: N={}, Err={}, Del={}, Ins={}, Sub={}'.format(*word_rec))
    per = list(map(lambda x: f'{(x / word_rec[0]) * 100:.2f}%', word_rec[1:]))
    print('Avg: N={}, Err={}, Del={}, Ins={}, Sub={}'.format(*[word_rec[0], *per]))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate ASR results.\n"
                                                 "Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument('-ht', '--hyptrn', help='Hypothesized transcripts in TRN format', required=True, default=None)
    parser.add_argument('-rt', '--reftrn', help='Reference transcripts in TRN format', required=True, default=None)
    args = parser.parse_args()

    if args.reftrn is None or args.hyptrn is None:
        RuntimeError("Must specify reference trn and hypothesis trn files.")

    score(ref_trn=args.reftrn, hyp_trn=args.hyptrn)

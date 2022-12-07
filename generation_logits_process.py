# transformers generation_logits_process.py에 추가할 항목

# history 반복하지 않게끔
class HistoryNoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, history_ngram_size: int, history_input_ids: torch.LongTensor):
        if not isinstance(history_ngram_size, int) or history_ngram_size <= 0:
            raise ValueError(
                f"`history_ngram_size` has to be a strictly positive integer, but is {history_ngram_size}"
            )
        self.ngram_size = history_ngram_size
        if len(history_input_ids.shape) == 1:
            history_input_ids = history_input_ids.unsqueeze(0)
        self.batch_size = history_input_ids.shape[0]
        self.generated_ngrams = _get_ngrams(history_ngram_size, history_input_ids, self.batch_size)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # B x num_beams
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = [
            _get_generated_ngrams(
                self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len
            )
            for hypo_idx in range(num_hypos)
        ]

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores

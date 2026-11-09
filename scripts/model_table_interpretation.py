"""Insert prose interpretations after result tables without changing measured data."""
from __future__ import annotations

from collections import Counter


def annotate_tables(markdown: str, notes: list[str]) -> str:
    """Require one explicit interpretation for each existing Markdown table."""
    lines = markdown.splitlines()
    rendered = []
    index = 0
    for offset, line in enumerate(lines):
        rendered.append(line)
        if line.startswith('|') and (offset + 1 == len(lines) or not lines[offset + 1].startswith('|')):
            if index >= len(notes):
                raise ValueError('A result table is missing its interpretation')
            rendered.extend(['', notes[index]])
            index += 1
    if index != len(notes):
        raise ValueError(f'Expected {len(notes)} result tables, found {index}')
    return '\n'.join(rendered) + '\n'


def task_interpretation(data: dict, tier: str) -> str:
    rows = [r for r in data['runs'] if r['vocab'] == tier and r['model_step'] is not None]
    names = {'none': 'raw BPE', 'TRmorph': 'TRmorph', 'Zemberek': 'Zemberek', 'TurkishDelightNLP': 'TurkishDelight'}
    wins = Counter()
    ties = []
    for task in data['task_metrics']:
        if task['task'] == 'xquad_tr':
            continue
        best = max(task['values'][r['run_id']] for r in rows)
        leaders = [r for r in rows if task['values'][r['run_id']] == best]
        if len(leaders) == 1:
            wins[leaders[0]['segmenter']] += 1
        else:
            tie_names = 'all four tokenizers' if len(leaders) == 4 else ' and '.join(names[r['segmenter']] for r in leaders)
            ties.append(f"`{task['task']}` ({tie_names})")
    counts = ', '.join(f"{names[r['segmenter']]} {wins[r['segmenter']]}" for r in rows)
    assert sum(wins.values()) + len(ties) == 11
    return (
        f'**Interpretation.** Across the eleven classification tasks, outright first-place counts at the displayed precision are {counts}; '
        + ('the tied first places are ' + '; '.join(ties) + '. ' if ties else 'there are no tied first places. ')
        + 'These counts describe the recorded table, while the macro weights score magnitudes rather than task wins. '
        + 'The legacy continuation-boundary error makes every ranking provisional pending a corrected common rerun. '
        + 'XQuAD is excluded from these counts because its legacy generation bug prevents a tokenizer-benefit conclusion.'
    )


def model_notes(data: dict) -> list[str]:
    return [
        '**Reading note.** Each vocabulary tier keeps the training horizon and approximate parameter budget fixed, while depth falls from 20 to 16 to 12. This supports comparisons among tokenizers inside a tier. A comparison across tiers changes both vocabulary and model architecture, so it cannot identify the effect of vocabulary size alone.',
        '**Interpretation.** Raw BPE records the lowest validation BPB at every tier: 0.6232, 0.6409 and 0.6749. The closest MorphBPE alternatives are Zemberek at 32k and TurkishDelight at 64k/128k, with BPB gaps of 0.0018, 0.0101 and 0.0071. Lower MorphBPE token-level losses do not reverse that ordering because token losses use different prediction units. BPB comparisons still depend on the unmatched raw-text evaluation-span caveat; the absent 32k TurkishDelight model has no ranking.',
        '**Interpretation.** The shortest recorded training-loop time within each tier belongs to TRmorph at 32k, TurkishDelight at 64k and Zemberek at 128k. Differences among tokenizers in the same tier are small, whereas the shallower 64k/128k models need much less time, estimated compute and peak memory than d20. Those larger cross-tier savings combine architecture and vocabulary changes; a single timing run does not establish a tokenizer speed advantage.',
        '**Interpretation.** The recorded core-11 leader is Zemberek at 32k (0.4618), then raw BPE at 64k (0.4590) and 128k (0.4651). Thus the legacy aggregate contains no consistent MorphBPE winner. The likelihood-boundary error requires a corrected rerun before treating those leads as model-quality or tokenizer effects, and the generation bug prevents interpreting the XQuAD leads as tokenizer gains. CETVEL throughput describes the entire legacy harness, with depth also changing across tiers.',
        task_interpretation(data, '32k'),
        task_interpretation(data, '64k'),
        task_interpretation(data, '128k'),
        '**Reading note.** This historical table contains only the raw-BPE baseline, so it cannot by itself rank tokenizers. Additional check-worthiness and relevance-judgment submetrics for the other models are preserved in the all-metric CSV. Accuracy and normalized accuracy are different scoring rules; the TQuAD exact/F1 values also retain the legacy generation limitation.',
        '**Interpretation.** The higher-precision metadata preserves the same BPB ordering: Zemberek is the better of the two recovered 32k MorphBPE models, and raw BPE leads the recovered 64k/128k rows. The missing raw-BPE32k metadata cells do not erase its archived 0.6232 result. Shorter training counters at larger vocabularies coincide with fewer layers and narrower hidden states, so this table is not an isolated vocabulary ablation.',
        '**Interpretation.** The fastest final-100-step median is TRmorph within 32k, TurkishDelight within 64k and Zemberek within 128k. Each lead over its raw-BPE baseline is under 1%, and memory use is nearly unchanged within a tier. The large speed increase from d20 to d12 comes with a different architecture; higher d20 MFU does not imply higher tokens/s or shorter training.',
        '**Interpretation.** Raw BPE also has the lowest standalone validation BPB within all three tiers, so the separate evaluation call agrees with the checkpoint BPB ordering. Small last-decimal changes between the two validation calls are evaluation differences, not evidence of another training improvement. Training-set BPB and token-level loss should not substitute for the validation comparison.',
        '**Reading note.** This single d2 run demonstrates that tokenization, training, checkpointing and evaluation completed on the cluster. Its short context, small model and separate tokenizer-training cap make its time, MFU and BPB unsuitable for ranking the d20/d16/d12 treatments.',
    ]


CATEGORY_NOTE = (
    '**Interpretation.** The d32 receipt has its highest category score in summarization (81.0680) and its lowest in grammatical correction (0.9678), with QA also low at 8.3129. '
    'The categories use different metrics and task difficulties, so their numerical order is a profile of this protocol rather than a common measure of capability. '
    'The 37.1709 overall score has no same-protocol competing model here and cannot establish a tokenizer or model-size winner.'
)
TASK_NOTE = (
    '**Interpretation.** This table describes one d32 model, so it has no cross-model winner. Among its raw-accuracy tasks, news classification (70.0) and XCOPA (69.0) have higher recorded accuracy than TurkishMMLU (18.22) and proverbs (0.0), although these tasks differ in difficulty and chance level. '
    'The high summarization BERTScores and low QA exact-match/GEC scores use different definitions; they must not be ranked together as if they measured the same thing. '
    'A same-protocol baseline is needed to turn this task profile into a configuration comparison.'
)
QA_NOTE = (
    '**Interpretation.** Correcting the evaluator raises the unchanged d20 checkpoint from 3.0985 to 11.3369 XQuAD F1, so that increase comes from decoding/scoring behavior, not training or tokenization. '
    'With the corrected 128-token setup, d32 scores higher than d20 (15.8833 versus 11.3369), but both use raw BPE and also differ in model size and training exposure. '
    'The separate 256-token d32 result is lower (15.1717) under different recorded runner/protocol settings; this is not an isolated generation-cap experiment. None of these comparisons establishes a MorphBPE benefit.'
)
PRETRAIN_NOTE = (
    '**Interpretation.** Continuing d20 lowers its recorded BPB from 0.62321265 to 0.59145082 with twice the scheduled token budget and a changed learning-rate policy. '
    'There is no paired continuation benchmark to show whether that translates into downstream improvement. '
    'The larger d32 records 0.51903967 on training-overlap text, so that value cannot demonstrate a held-out advantage over d20 or any tokenizer treatment.'
)


def configuration_notes(data: dict) -> list[str]:
    # Count task directions from the preserved results rather than transcribing them.
    inventory = data['evaluation_inventory']
    baseline = next(r['results'] for r in inventory if r['protocol'].startswith('full_kvcache_'))
    scaled = next(r['results'] for r in inventory if r['protocol'].startswith('core11_legacy_'))
    tasks = ('exams_tr', 'belebele_tr', 'turkish_plu', 'cetvel_xcopa_tr', 'cetvel_xnli_tr', 'mnli_tr', 'snli_tr', 'news_cat', 'offenseval_tr', 'trclaim19', 'xfact_tr')
    differences = [scaled[t]['acc,none' if t == 'cetvel_xcopa_tr' else 'acc_norm,none'] - baseline[t]['acc,none' if t == 'cetvel_xcopa_tr' else 'acc_norm,none'] for t in tasks]
    higher = sum(value > 0 for value in differences)
    lower = sum(value < 0 for value in differences)
    tied = sum(value == 0 for value in differences)
    assert higher + lower + tied == 11
    direction_note = f'It has higher recorded scores on {higher} of the eleven tasks and lower scores on {lower}' + (f', with {tied} ties' if tied else '') + ', with Offenseval contributing the largest drop. '
    return [
        PRETRAIN_NOTE,
        '**Interpretation.** On eight A100s, d32 has a much longer measured step and substantially larger peak memory than the d20 continuation; its higher MFU does not make it the cheaper configuration. The throughput columns use different units, so their ratio is not a precise speedup. Job duration also includes different training phases: d20 retains a usable final checkpoint despite a failed scheduler exit, while d32 finished cleanly.',
        '**Interpretation.** The larger d32 has a lower recorded legacy core-11 mean (0.4180 versus 0.4514), so these observations do not show a universal gain from scaling. ' + direction_note + 'This contrast remains provisional because the legacy likelihood scorer is flawed, and size, data eligibility and training budget all change together.',
        QA_NOTE,
        CATEGORY_NOTE,
        TASK_NOTE,
        '**Interpretation.** These are alternative metrics for the same d32 model, not competing tokenizer/configuration rows. Its recorded TQuAD overlap F1 is higher than XQuAD F1, but the tasks differ; translation WER and BLEU likewise measure different aspects of output. ROUGE, QA F1, exact match and the later BERTScore/COMET receipt must retain their native scales and protocols rather than being combined into a new ranking.',
        '**Reading note.** The inventory distinguishes saved results, missing checkpoints, malformed files and smoke checks. Missing entries are not zero scores, and a two-example smoke result is not a capability estimate. The available continuation/d32/SFT evidence broadens the configuration record without filling the missing matched 32k TurkishDelight cell.',
    ]


def later_notes() -> list[str]:
    return [QA_NOTE, PRETRAIN_NOTE, CATEGORY_NOTE, TASK_NOTE]


def sft_notes() -> list[str]:
    return [
        '**Reading note.** These curricula have different training sizes and, for calibration, a different validation set. The inventory identifies what each screen measured; larger datasets and later checkpoint numbers are not evidence of better behavior. Repair/calibration/polish start from the retained SFT source, while the combined SmolTalk2 checkpoint is a separate run from the base model.',
        '**Interpretation.** The high-rate arm at step 24 gives the lowest BPB (0.441730), but its probe score is only 6/41. The low-rate steps 8 and 16 lead the candidates at 8/41, still below the retained source at 9/41, and none passes the full selection gates. Here a lower validation loss does not identify the behaviorally better checkpoint.',
        '**Interpretation.** The strongest-rate arm gives the lowest BPB at step 8 (0.456307) and the highest probe count at step 12 (10/41). Although 10/41 exceeds the source’s 9/41 overall count, the candidate fails the required category/selection gates, so the decision remains retain source. Extra steps and a higher rate therefore do not establish an acceptable improvement on this screen.',
        '**Interpretation.** Within eff020, additional updates reduce validation BPB from 0.575542 at step 8 to 0.105042 at step 32. At step 8, eff040 has lower BPB than eff020. These are loss-only observations: the imported receipts do not show a passing behavior probe, and the incomplete eff040 trajectory prevents a full rate-by-horizon comparison. The lowest BPB row is not a selected best model.',
        '**Interpretation.** The retained source wins both behavior suites: 9 versus 5 passes on the probe and 68 versus 36 on the production suite. The candidate reduces prompt-echo counts in the production suite but increases repetition and loses overall passes, so its selection rejection follows the measured behavior. This is a post-training configuration result on one tokenizer, not a tokenizer-family comparison.',
    ]

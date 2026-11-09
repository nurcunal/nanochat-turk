"""Build descriptive SFT screen tables from immutable, hash-verified UHeM receipts."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'artifacts/uhem_sft_screens_2026-11-09'
OUT = ROOT / 'docs/study_results'


def main():
    manifest = json.loads((ARTIFACT / 'manifest.json').read_text())
    by_source = {}
    sources = []
    for entry in manifest['records']:
        local = ARTIFACT / entry['local_path']
        raw = local.read_bytes()
        assert len(raw) == entry['size_bytes']
        assert hashlib.sha256(raw).hexdigest() == entry['sha256'], local
        sources.append(entry)
        if raw:
            by_source[entry['path']] = json.loads(raw)
    def get(suffix):
        values = [v for k, v in by_source.items() if k.endswith(suffix)]
        assert len(values) == 1, suffix
        return values[0]
    def link(suffix, label='source'):
        entries = [e for e in sources if e['path'].endswith(suffix)]
        assert len(entries) == 1, suffix
        return f"[{label}](../../artifacts/uhem_sft_screens_2026-11-09/{entries[0]['local_path']})"
    rows = []
    text = [
        '# SFT configuration screens: d32 with raw BPE 32k',
        '',
        'These recovered posttraining results hold the tokenizer at **raw BPE, 32,768 tokens** and vary the training curriculum, learning rates and checkpoint. They supplement the [base-model tokenizer ablation](model_results.md); they are not additional tokenizer-family comparisons. All are descendants of the d32 FineWeb-2 model. The 41-case probe and 512-case production behavior suite are custom selection diagnostics, not CETVEL.',
        '',
        f"Read-only UHeM recovery: `{manifest['retrieved_at_utc']}`. [Source manifest](../../artifacts/uhem_sft_screens_2026-11-09/manifest.json), [flat CSV](sft_configuration_screens.csv), [full derived JSON](sft_configuration_screens.json). Original receipt bytes, source paths, SHA-256 hashes, code identities and checkpoint hashes are preserved. These summaries establish recorded results; this import does not re-execute their evaluators or verify current weight-file availability.",
        '',
        'The common retained SFT source is `tr_d32_bpe32k_smoltalk2_turk_full_v9_then_fast_stage_b_v1`, checkpoint **84**. Its custom probe score is **9/41**. Low validation BPB on these small SFT corpora does not establish better general language modeling or downstream capability. Compare BPB only within a dataset and scoring protocol; it is not comparable to pretraining BPB or across the different curricula below.',
        '',
        '## Dataset and protocol inventory',
        '',
        '| Study | Training rows | Validation rows | Evaluation | Status |',
        '| --- | ---: | ---: | --- | --- |',
    ]
    targeted_manifest = get('nanochat-turk-d32-targeted-repair-v3/manifest.json')
    calibration_manifest = get('nanochat-turk-d32-capability-calibration-v1/manifest.json')
    polish_ckpt = get('final-polish-v1/checkpoint-summary.json')
    combined_ckpt = get('final-combined-smoltalk2-turk-sft-v1/checkpoint-summary.json')
    for name, data, evaluation, status in [
        ('Targeted repair v3', targeted_manifest['outputs'], '41-case selection probe', 'Nine verified checkpoints; retain source'),
        ('Capability calibration v1', calibration_manifest['outputs'], 'Checkpoint validation BPB', 'Five verified checkpoints; incomplete screen'),
        ('Final polish v1', polish_ckpt['data'], '41-case probe + 512-case production suite', 'Checkpoint complete; rejected by selection gates'),
        ('Combined SmolTalk2-Turk SFT', combined_ckpt['data'], 'Completion receipt only', 'Checkpoint 139 complete; no score inferred'),
    ]:
        text.append(f"| {name} | {data['train']['rows']:,} | {data['validation']['rows']:,} | {evaluation} | {status} |")
    text += ['', 'Targeted repair and final polish share the exact 1,557-row validation file (SHA-256 `2cae224e5cbfa297c8b6e8ebd3226bd99ed426e876a936053bdc1f670b8cfb6d`). Capability calibration uses a different 400-row set. Both curriculum manifests specify a 2,048-token maximum. Bounded-recovery data-row counts are not present in the selected receipts and are left unspecified.', '', '## Bounded recovery: nine learning-rate/checkpoint candidates', '', 'The recovered selection reuses completed training and probe outputs after the initial selection step failed. The repaired decision is **retain source**: none of the nine candidates passes its selection gates. The arm names below are the recorded low/mid/high labels; exact checkpoint identities are retained in JSON.', '', '| Arm | Checkpoint | SFT validation BPB ↓ | Probe passes / 41 ↑ | Qualified |', '| --- | ---: | ---: | ---: | --- |']
    bounded = get('bounded-recovery-v1r2/probe-selection-summary.json')
    admission = get('bounded-recovery-v1r2/recovery-admission-summary.json')
    verify = admission['parent']['checkpoint_verifications']
    for arm in ('low', 'mid', 'high'):
        for step in (8, 16, 24):
            label = f'{arm}-{step}'
            candidate = bounded['candidates'][label]
            checkpoint = verify[f'arm-{arm}/checkpoint-{step}-verification.json']['checkpoint']
            assert checkpoint['step'] == step and candidate['total'] == 41
            row = {'study': 'bounded_recovery', 'label': label, 'tokenizer': 'bpe_32768', 'model_tag': checkpoint['model_tag'], 'step': step, 'validation_bpb': checkpoint['validation_bpb'], 'probe_passed': candidate['passed'], 'probe_total': candidate['total'], 'qualified': candidate['probe_qualified']}
            rows.append(row)
            text.append(f"| {arm} | {step} | {row['validation_bpb']:.6f} | {row['probe_passed']}/41 | {'Yes' if row['qualified'] else 'No'} |")
    text += ['', f"Evidence: {link('bounded-recovery-v1r2/probe-selection-summary.json', 'probe scores')}, {link('bounded-recovery-v1r2/recovery-admission-summary.json', 'checkpoint BPB and hashes')}, {link('bounded-recovery-v1r2/recovery-decision.json', 'final decision')}.", '', '## Targeted repair: learning rates and checkpoint horizon', '', 'The table reports the declared matrix learning rate and the receipt’s effective initial smear learning rate. Other optimizer-group rates and the initial multiplier are preserved in the source receipts. All nine rows are diagnostic selection candidates; the recorded decision is **retain source**, with no probe-qualified candidate.', '', '| Arm | Step | Matrix LR | Initial smear LR | SFT validation BPB ↓ | Probe passes / 41 ↑ | Qualified |', '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    targeted = get('targeted-repair-screen-v3-probe-recovery-v1/probe-selection.json')
    for arm in ('eff005', 'eff010', 'eff020'):
        for step in (4, 8, 12):
            label = f'{arm}-{step}'
            v = get(f'targeted-repair-screen-v3-probe-recovery-v1/{label}-checkpoint-verification-summary.json')
            candidate = targeted['candidates'][label]
            assert v['status'] == 'pass' and candidate['total'] == 41
            lr = v['optimizer_lr_identity']
            row = {'study': 'targeted_repair_v3', 'label': label, 'tokenizer': 'bpe_32768', 'model_tag': v['checkpoint']['model_tag'], 'step': step, 'matrix_lr': lr['declared']['matrix_lr'], 'effective_smear_initial_lr': lr['effective_smear_initial_lr'], 'validation_bpb': v['validation_bpb'], 'probe_passed': candidate['passed'], 'probe_total': candidate['total'], 'qualified': candidate['probe_qualified']}
            rows.append(row)
            text.append(f"| {arm} | {step} | {row['matrix_lr']:.4f} | {row['effective_smear_initial_lr']:.4f} | {row['validation_bpb']:.6f} | {row['probe_passed']}/41 | {'Yes' if row['qualified'] else 'No'} |")
    text += ['', f"Evidence: {link('targeted-repair-screen-v3-probe-recovery-v1/probe-selection.json', 'probe selection')}, {link('targeted-repair-screen-v3-probe-recovery-v1/screen-decision-summary.json', 'decision')}. Each checkpoint verifier is linked through the source manifest.", '', '## Capability calibration: completed checkpoints', '', 'This is a separate 7,600-training/400-validation-row curriculum. These rows establish checkpoint verification and its recorded validation BPB. They do not establish passing behavior probes or a selected release. The `eff040-16` summary is empty; no value is inferred for it or other unreported arms.', '', '| Arm | Step | Matrix LR | Initial smear LR | SFT validation BPB ↓ | Verification |', '| --- | ---: | ---: | ---: | ---: | --- |']
    calibration_rows = [(path, v) for path, v in by_source.items() if '/capability-calibration-screen-v1/' in path and path.endswith('-checkpoint-verification-summary.json')]
    for path, v in sorted(calibration_rows, key=lambda item: (item[1]['label'].split('-')[0], item[1]['checkpoint']['step'])):
        assert v['status'] == 'pass'
        lr = v['optimizer_lr_identity']; checkpoint = v['checkpoint']
        row = {'study': 'capability_calibration_v1', 'label': v['label'], 'tokenizer': 'bpe_32768', 'model_tag': checkpoint['model_tag'], 'step': checkpoint['step'], 'matrix_lr': lr['declared']['matrix_lr'], 'effective_smear_initial_lr': lr['effective_smear_initial_lr'], 'validation_bpb': v['validation_bpb'], 'verification_status': v['status']}
        rows.append(row)
        text.append(f"| {v['label'].split('-')[0]} | {row['step']} | {row['matrix_lr']:.4f} | {row['effective_smear_initial_lr']:.4f} | {row['validation_bpb']:.6f} | Pass |")
    text += ['', '## Final polish: source versus candidate', '', 'The candidate is `tr_d32_bpe32k_final_instruct_polish_v1`, checkpoint **64**. Both checkpoints use raw BPE 32k. The saved selection rejects the candidate and retains source checkpoint 84. Pass counts below are custom behavior checks. Diagnostics such as echo and repetition are independent counts and need not sum to the suite size.', '', '| Model | Suite | Passes ↑ | Pass rate ↑ | Empty ↓ | Echo ↓ | Repetition ↓ | Assistant-end termination |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    polish = get('final-polish-v1/selection-summary.json')
    for label in ('source', 'candidate'):
        for suite in ('probe', 'production'):
            score = polish[label][suite]; diag = score['diagnostics']
            row = {'study': 'final_polish_v1', 'label': label, 'tokenizer': 'bpe_32768', 'model_tag': polish_ckpt['source_checkpoint']['model_tag'] if label == 'source' else polish_ckpt['model_tag'], 'step': 84 if label == 'source' else 64, 'suite': suite, 'passed': score['passed'], 'total': score['total'], 'pass_rate': score['passed']/score['total'], **diag}
            rows.append(row)
            text.append(f"| {label.title()} | {suite} | {row['passed']}/{row['total']} | {row['pass_rate']:.2%} | {diag['empty_completion_count']} | {diag['prompt_echo_count']} | {diag['repetition_count']} | {diag['assistant_end_termination_count']} |")
    text += ['', f"Evidence: {link('final-polish-v1/selection-summary.json', 'selection receipt')} and {link('final-polish-v1/checkpoint-summary.json', 'checkpoint/data receipt')}.", '', '## Completed checkpoint without an imported benchmark', '', f"`{combined_ckpt['model_tag']}` checkpoint **{combined_ckpt['step']}** has a completed receipt, one-rank optimizer inventory, and training/validation manifest hashes. This establishes the saved SFT checkpoint, not a measured improvement. No benchmark value is inferred from completion. Evidence: {link('final-combined-smoltalk2-turk-sft-v1/checkpoint-summary.json', 'completion receipt')}.", '', '## Coverage and limitations', '', '- These are selected recoverable configuration screens from UHeM, not a claim that every operational pilot or intermediate checkpoint has been exhaustively audited.', '- Repeated checks of the same checkpoint and selection recovery do not count as independent training runs or seed repetitions.', '- The initial bounded-recovery selection summary and capability `eff040-16` summary were empty at recovery; both are recorded as unavailable in the manifest. Their missing values are not zero.', '- No matched multi-tokenizer SFT comparison was recovered. These d32/raw-BPE32k screens stay separate from the eleven-model tokenizer comparison and official CETVEL scores.', '- Rebuild with `python -m scripts.build_sft_configuration_results`. The builder verifies all original-file SHA-256 values, checkpoint steps, probe denominators and verification statuses before emitting the tables.', '']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sft_configuration_screens.md').write_text('\n'.join(text))
    (OUT / 'sft_configuration_screens.json').write_text(json.dumps({'retrieved_at_utc':manifest['retrieved_at_utc'], 'scope':'d32/BPE32k custom SFT configuration diagnostics', 'rows':rows, 'sources':sources, 'records':by_source}, indent=2, ensure_ascii=False)+'\n')
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (OUT / 'sft_configuration_screens.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    assert len(rows) == 27
    print(f'Wrote {len(rows)} configuration/probe rows from {len(sources)} hash-verified source files')


if __name__ == '__main__':
    main()

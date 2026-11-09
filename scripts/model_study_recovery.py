"""Add recovered UHeM measurements without replacing historical score records."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from statistics import mean

from scripts.study_report_paths import write_section

EVIDENCE = 'artifacts/model_evidence_2026-11-09'
NAMES = {'none': 'Raw BPE', 'TRmorph': 'MorphBPE / TRmorph', 'Zemberek': 'MorphBPE / Zemberek', 'TurkishDelightNLP': 'MorphBPE / TurkishDelight'}


def dump_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def number(v, digits=4):
    return '—' if v is None else f'{v:.{digits}f}'


def enrich(root, out, data):
    source_paths = [f'{EVIDENCE}/model-metadata-and-benchmarks.json', f'{EVIDENCE}/later-model-metadata-and-benchmarks.json']
    recovered, later = [json.loads((root / p).read_text()) for p in source_paths]
    data['sources'] += [{'path': p, 'sha256': hashlib.sha256((root / p).read_bytes()).hexdigest()} for p in source_paths]
    data['recovery'] = {'retrieved_at_utc': recovered['retrieved_at_utc'], 'host': recovered['host'], 'source_artifact': source_paths[0], 'preserves_original_archived_values': True}
    data['recovery_limitations'] = [
        'Historical rounded scores remain in runs and task_metrics. Recovered full-precision scores, all metrics, and per-task standard errors are additional evidence.',
        'The original raw-BPE d20 final metadata path was absent at recovery. Its archived BPB remains available; no checkpoint was inferred from a matching directory name.',
        'Training time is the checkpoint loop_state.total_training_time counter, not scheduler wall time. The counter excludes initial compile/warmup steps as well as evaluation/checkpoint/setup overhead. Use the separately logged last-100-step rate for throughput; do not divide the full training budget by this partial-time counter.',
        'Legacy-generation XQuAD scores remain grouped together. Fixed-generation evaluations and EACL table-compatible scoring are separate protocols.',
        'Legacy classification/log-likelihood scores may be affected by the unvalidated context/continuation token boundary; their rankings need a corrected common rerun.',
    ]
    all_task_rows = []
    performance = []
    for r in recovered['records']:
        row = next(x for x in data['runs'] if x.get('run_id') == r['run_id'])
        meta_record = r['source_meta_path']
        meta = meta_record.get('data')
        result_record = r['source_results_path']
        results = result_record['data']
        for task in data['task_metrics']:
            precise = results['results'][task['task']][task['metric']]
            assert round(precise, 4) == task['values'][r['run_id']], (r['run_id'], task['task'])
        primary = {t['task']: results['results'][t['task']][t['metric']] for t in data['task_metrics']}
        precise_macro = mean(primary[t] for t in data['macro_core11_tasks'])
        assert round(precise_macro, 4) == row['macro_core11'], (r['run_id'], precise_macro, row['macro_core11'])
        row['recovered'] = {
            'source_artifact': source_paths[0], 'checkpoint_metadata_available': bool(meta),
            'meta_provenance': {k: v for k, v in meta_record.items() if k != 'data'},
            'evaluation_provenance': {k: v for k, v in result_record.items() if k != 'data'},
            'full_precision_primary_task_scores': primary,
            'full_precision_core11_mean': precise_macro,
            'all_task_metrics': results['results'], 'evaluation_config': results['config'],
            'sample_counts': results['n-samples'],
        }
        perf = {'run_id': r['run_id'], 'tokenizer': row['tokenizer'], 'vocab_size': row['vocab_size'], 'depth': row['transformer_depth'], 'model_step': row['model_step'], 'training_token_positions': row['training_token_positions'], 'checkpoint_metadata_available': bool(meta), 'source_meta_path': meta_record['path']}
        if meta:
            assert meta['step'] == row['model_step']
            assert round(meta['val_bpb'], 4) == row['validation_bpb']
            assert meta['val_bpb'] == meta['loop_state']['min_val_bpb']
            config, user, loop = meta['model_config'], meta['user_config'], meta['loop_state']
            assert (config['n_layer'], config['vocab_size']) == (row['transformer_depth'], row['vocab_size'])
            assert meta['total_batch_size'] == 1048576 and config['sequence_len'] == 2048
            elapsed = loop['total_training_time']
            row['recovered'].update(validation_bpb=meta['val_bpb'], minimum_validation_bpb=loop['min_val_bpb'], smooth_train_loss=loop['smooth_train_loss'], model_config=config, user_config=user, training_loop_seconds=elapsed)
            perf.update(width=config['n_embd'], heads=config['n_head'], kv_heads=config['n_kv_head'], context=config['sequence_len'], attention_pattern=config['window_pattern'], fp8=user['fp8'], device_batch_sequences=meta['device_batch_size'], global_batch_token_positions=meta['total_batch_size'], validation_bpb=meta['val_bpb'], smooth_train_loss=loop['smooth_train_loss'], training_loop_seconds=elapsed, training_loop_hours=elapsed / 3600)
        performance.append(perf)
        for task, metrics in results['results'].items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    all_task_rows.append({'run_id': row['run_id'], 'tokenizer': row['tokenizer'], 'protocol': 'legacy_core12_and_available_subtasks', 'task': task, 'metric': metric, 'value': value, 'effective_examples': results.get('n-samples', {}).get(task, {}).get('effective'), 'source_path': result_record['path'], 'source_sha256': result_record['sha256']})
    enrich_training_logs(root, data, performance)
    data['recovered_training_performance'] = performance
    data['audit']['recovered_primary_task_cells_verified'] = 132
    data['audit']['recovered_checkpoint_metadata_count'] = sum(p['checkpoint_metadata_available'] for p in performance)
    data['audit']['recovered_full_precision_macros_match_archive'] = True
    dump_csv(out / 'model_training_performance.csv', performance)
    later_perf_path = f'{EVIDENCE}/later-training-performance.json'
    data['sources'].append({'path': later_perf_path, 'sha256': hashlib.sha256((root / later_perf_path).read_bytes()).hexdigest()})
    later_report = later_models(root, out, later, source_paths[1], all_task_rows)
    data['configuration_experiments'] = later_report
    dump_csv(out / 'model_all_task_metrics.csv', all_task_rows)
    return recovered_markdown(data)


def enrich_training_logs(root, data, performance):
    path = f'{EVIDENCE}/training-performance.json'
    bundle = json.loads((root / path).read_text())
    data['sources'].append({'path': path, 'sha256': hashlib.sha256((root / path).read_bytes()).hexdigest()})
    for record in bundle['records']:
        p = next(x for x in performance if x['run_id'] == record['run_id'])
        run = next(x for x in data['runs'] if x.get('run_id') == record['run_id'])
        lines = '\n'.join(x['text'] for x in record['excerpts'])
        parse = lambda pattern: re.search(pattern, lines, re.M).group(1)
        logged_minutes = float(parse(r'^Total training time: ([\d.]+)m'))
        total_parameters = int(parse(r'^total\s+: ([\d,]+)').replace(',', ''))
        checkpoint_bpb = float(parse(r'^Step 17100 \| Validation bpb: ([\d.]+)'))
        assert round(checkpoint_bpb, 4) == run['validation_bpb']
        assert record['last_100_printed_steps']['count'] == 100
        assert record['last_100_printed_steps']['start_step'] == 17000
        assert record['last_100_printed_steps']['end_step'] == 17099
        if 'training_loop_seconds' in p:
            assert abs(p['training_loop_seconds'] / 60 - logged_minutes) <= 0.01
        p.update(total_parameters=total_parameters, logged_training_minutes=logged_minutes,
            logged_checkpoint_validation_bpb=checkpoint_bpb,
            peak_memory_mib=float(parse(r'^Peak memory usage: ([\d.]+)MiB')),
            standalone_base_eval_train_bpb=float(parse(r'^train bpb: ([\d.]+)')),
            standalone_base_eval_validation_bpb=float(parse(r'^val bpb: ([\d.]+)')),
            slurm_job_id=int(parse(r'^Job ID: (\d+)')),
            slurm_wall_clock=parse(r'^Job Wall-clock time: (.+)$'),
            median_last100_tokens_per_second=record['last_100_printed_steps']['median_tokens_per_second'],
            median_last100_dt_ms=record['last_100_printed_steps']['median_dt_ms'],
            median_last100_bf16_mfu_percent=record['last_100_printed_steps']['median_bf16_mfu_pct'],
            source_training_log=record['path'], source_training_log_sha256=record['sha256'])
        run['recovered_training_log'] = record
        run['total_parameters'] = total_parameters
    data['audit']['training_logs_verified'] = len(bundle['records'])


def recovered_markdown(data):
    rows = []
    for p in data['recovered_training_performance']:
        rows.append([p['tokenizer'], p['depth'], p.get('width', '—'), p.get('heads', '—'), number(p.get('validation_bpb'), 8), number(p.get('training_loop_hours'), 3)])
    text = ['\n## Recovered checkpoint configuration and training performance\n',
        'The compact [UHeM evidence bundle](../../' + EVIDENCE + '/) was recovered on 9 November 2026. It supplies ten of the eleven final checkpoint metadata records and full-precision evaluation results for all eleven models. All **132 primary task cells** reproduce the archived four-decimal values; all recovered core-11 means reproduce their archived macros. The additional [all-metric CSV](model_all_task_metrics.csv) includes available auxiliary metrics and per-task standard errors, not just the original selected metric.\n',
        'The table reports `loop_state.total_training_time`, an accumulated training-loop counter that excludes initial compile/warmup steps, evaluation, checkpoint and setup overhead. It is not scheduler wall-clock time. Throughput below uses logged step measurements; the full position budget is not divided by this partial-time counter. [CSV](model_training_performance.csv) and [JSON](model_results.json) retain exact values, model and optimizer settings, source paths and hashes.\n',
        table(['Tokenizer', 'Layers ↔', 'Width ↔', 'Heads ↔', 'Recovered BPB ↓', 'Training-loop h ↓'], rows),
        '\nAll ten recovered configurations use context 2,048, full attention (`L`), matching query/KV heads, global batch 1,048,576 token positions, per-device batch 4, and FP8 disabled. The original raw-BPE d20 final metadata path was absent at recovery; its archived BPB and existing benchmark remain valid historical evidence. A dash means the value is unavailable from this metadata source.\n',
        'The [additional model-configuration report](model_configuration_results.md) contains the d20 continuation, completed d32 FineWeb-2 model, fixed-generation comparisons and full EACL table-compatible benchmark. It explicitly separates protocols and the d32 training-overlap BPB diagnostic from the controlled tokenizer ablation.\n']
    text += [
        '\n### Training throughput, memory and scheduler time\n',
        'Fresh log recovery preserves all eleven completed runs. Tokens/s, step time and BF16 MFU are medians of printed steps 17,000–17,099 (100 steps per run); MFU follows the logger’s hardware/FLOP convention. Peak MiB is the logged training-process memory figure, not total cluster RAM. Scheduler wall time includes other job work and is distinct from the training-loop counter.\n',
        'Direction labels assume matched workload and hardware: tokens/s ↑ is throughput, step time and elapsed time ↓ are latency/cost, and peak memory ↓ is resource demand. MFU ↔ is estimated hardware-compute utilization, not a model-quality score. Parameters ↔ describe model size. See the [metric guide](metric_guide.md).\n',
        table(['Tokenizer','Parameters ↔','Training min ↓','Slurm wall time ↓','Median tok/s ↑','Median step ms ↓','BF16 MFU % ↔','Peak MiB ↓'], [[p['tokenizer'],f"{p['total_parameters']:,}",number(p['logged_training_minutes'],2),p['slurm_wall_clock'],number(p['median_last100_tokens_per_second'],1),number(p['median_last100_dt_ms'],2),number(p['median_last100_bf16_mfu_percent'],2),number(p['peak_memory_mib'],2)] for p in data['recovered_training_performance']]),
        '\n### Standalone base-evaluation BPB from training logs\n',
        'These logged train/validation evaluations run after the final checkpoint. They are a separate evaluation call and are preserved separately from checkpoint validation BPB; they must not silently replace the earlier measurement.\n',
        table(['Tokenizer','Checkpoint val BPB (log) ↓','Base-eval train BPB ↓','Base-eval val BPB ↓'], [[p['tokenizer'],number(p['logged_checkpoint_validation_bpb'],6),number(p['standalone_base_eval_train_bpb'],6),number(p['standalone_base_eval_validation_bpb'],6)] for p in data['recovered_training_performance']]),
        '\n### Archived d2 pipeline smoke performance\n',
        'This separate June 7 job validates the pipeline, not the controlled tokenizer comparison. Its tokenizer was trained on a 50-million-character cap and is named `bpe_32768_uhem_smoke`.\n',
        table(['Depth / width / heads ↔','Context ↔','Parameters ↔','Steps ↔','Training positions ↔','Training min ↓','Slurm wall time ↓','MFU % ↔','Peak MiB ↓'], [['2 / 128 / 2',256,'12,976,182',1980,'259,522,560','11.60','00:17:38','2.51','1,420.94']]),
        'The smoke artifact also records tokenizer-training time 6.3158 s, 7.347524e15 training FLOPs, 19.79 GB job memory utilized, and standalone train BPB 1.295173 / validation BPB 1.239951. Its checkpoint validation BPB is 1.239557.\n',
    ]
    return '\n'.join(text)


def later_models(root, out, bundle, source_path, all_task_rows):
    records = bundle['records']
    find = lambda substring: next(r for r in records if substring in r['path'])
    continuation = find('cont40x/meta_034200')
    d32 = find('fineweb2_all_v1/meta_028000')
    launch = find('/launch/tr_d32_fineweb2')
    receipt = find('official_scoring_receipt.json')
    verification = find('/production_verification.json')
    c, d, manifest, official = continuation['data'], d32['data'], launch['data'], receipt['data']
    assert official['complete'] is True
    assert official['model']['checkpoint_step'] == d['step'] == 28000
    assert math.isclose(mean(official['category_scores'].values()), official['overall_score'], abs_tol=1e-12)
    for cat, detail in official['category_details'].items():
        assert detail['complete'] and not detail['missing_tasks']
        assert math.isclose(mean(official['task_scores'][t] for t in detail['included_tasks']), official['category_scores'][cat], abs_tol=1e-12)
    rows = []
    for name, meta, record, count, scope in [
        ('d20 raw-BPE continuation', c, continuation, None, 'Continuation of d20 baseline; historical loader validation; distinct schedule'),
        ('d32 raw-BPE FineWeb-2', d, d32, manifest['model']['total_parameters'], 'TRAIN-OVERLAP diagnostic: validation source also eligible for optimization'),
    ]:
        rows.append({'experiment': name, 'tokenizer': meta['tokenizer_name'], 'depth': meta['model_config']['n_layer'], 'width': meta['model_config']['n_embd'], 'heads': meta['model_config']['n_head'], 'context': meta['model_config']['sequence_len'], 'total_parameters': count, 'checkpoint_step': meta['step'], 'global_batch_token_positions': meta['total_batch_size'], 'scheduled_cumulative_token_positions': meta['step'] * meta['total_batch_size'], 'validation_bpb': meta['val_bpb'], 'minimum_validation_bpb': meta['loop_state']['min_val_bpb'], 'smooth_train_loss': meta['loop_state']['smooth_train_loss'], 'training_loop_seconds': meta['loop_state']['total_training_time'], 'training_loop_hours': meta['loop_state']['total_training_time'] / 3600, 'validation_scope': scope, 'source_path': record['path'], 'source_sha256': record['sha256']})
    performance_path = f'{EVIDENCE}/later-training-performance.json'
    performance = json.loads((root / performance_path).read_text())
    for row, log_record in zip(rows, performance['records']):
        assert (row['depth'], log_record['run_id']) in {(20, 'd20_cont40x'), (32, 'd32_fineweb2')}
        lines = '\n'.join(x['text'] for x in log_record['excerpts'])
        parse = lambda pattern: re.search(pattern, lines, re.M).group(1)
        stats = log_record['last_100_printed_steps']
        assert stats['count'] == 100
        assert stats['end_step'] - stats['start_step'] + 1 == 100
        assert abs(row['training_loop_seconds'] / 60 - float(parse(r'^Total training time: ([\d.]+)m'))) < 0.01
        row.update(total_parameters=int(parse(r'^total\s+: ([\d,]+)').replace(',', '')),
            slurm_job_id=int(parse(r'^Job ID: (\d+)')),
            slurm_state=parse(r'^State: (.+)$'), slurm_wall_clock=parse(r'^Job Wall-clock time: (.+)$'),
            median_last100_throughput=stats['median_tokens_per_second'], throughput_units=log_record['throughput_units'],
            median_last100_dt_ms=stats['median_dt_ms'], median_last100_bf16_mfu_percent=stats['median_bf16_mfu_pct'],
            median_step_start=stats['start_step'], median_step_end=stats['end_step'],
            peak_memory_mib=float(parse(r'^Peak memory usage: ([\d.]+)MiB')),
            source_training_log=log_record['path'], source_training_log_sha256=log_record['sha256'])
    eval_inventory = []
    for record in records:
        values = record.get('data', {})
        if 'results' not in values:
            continue
        path = record['path']
        protocol = path.split('/cetvel_out/')[1].split('/')[0]
        smoke = values['config'].get('limit') is not None
        eval_inventory.append({'protocol': protocol, 'model_args': values['config']['model_args'], 'limit': values['config'].get('limit'), 'is_smoke': smoke, 'task_count_including_subgroups': len(values['results']), 'source_path': path, 'source_sha256': record['sha256'], 'results': values['results'], 'sample_counts': values['n-samples']})
        for task, metrics in values['results'].items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    all_task_rows.append({'run_id': 'd20' if ('reference_d20' in path or '/nanochat-turk-d20-' in path) else 'd32', 'tokenizer': 'bpe_32768', 'protocol': protocol, 'task': task, 'metric': metric, 'value': value, 'effective_examples': values['n-samples'].get(task, {}).get('effective'), 'source_path': path, 'source_sha256': record['sha256'], 'is_smoke': smoke})
    result = {'schema_version': 1, 'report_date': '2026-11-09', 'retrieved_at_utc': bundle['retrieved_at_utc'], 'source_artifact': source_path, 'source_sha256': hashlib.sha256((root / source_path).read_bytes()).hexdigest(), 'runs': rows, 'evaluation_inventory': eval_inventory, 'official_cetvel_receipt': official, 'official_verification': verification['data'], 'launch_manifest': manifest, 'checkpoint_metadata': [continuation, d32], 'training_performance_evidence': performance, 'training_performance_artifact': performance_path, 'training_performance_artifact_sha256': hashlib.sha256((root / performance_path).read_bytes()).hexdigest(), 'excluded_or_unreadable': [r for r in records if 'error' in r], 'limitations': ['d32 validation shard overlaps training and its BPB is not comparable to held-out BPB.', 'd20 continuation and d32 scale/budget changes are not tokenizer-controlled ablations.', 'Legacy classification/log-likelihood values may be affected by the unvalidated continuation token boundary; differences need a corrected common rerun.', 'Generation max-token budgets and evaluation protocols differ; keep result sets separate.', 'The repository table-compatible receipt is not an independent verification against an official public leaderboard.', 'The two smoke evaluations use two examples per task and are not capability estimates.', 'The continuation metadata total_training_time may include inherited training time; no new-phase tok/s is inferred.', 'No multi-tokenizer SFT or additional full 32k TurkishDelight model evidence is present in this recovery.']}
    (out / 'model_configuration_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    dump_csv(out / 'model_configuration_results.csv', rows)
    official_rows = [{'task': task, **detail} for task, detail in official['task_details'].items()]
    dump_csv(out / 'model_official_task_metrics.csv', official_rows)
    from scripts.model_table_interpretation import annotate_tables, configuration_notes
    write_section('model_configuration_results.md', highlight_configuration_tables(annotate_tables(later_markdown(result, records), configuration_notes(result))))
    return {'report': '../README.md#model-configurations', 'json': 'model_configuration_results.json', 'experiments': rows, 'official_overall_score': official['overall_score'], 'official_protocol': official['protocol']['id']}


def highlight_configuration_tables(text):
    from scripts.study_table_highlights import highlight_columns, highlight_rows, map_tables

    def transform(headers, rows):
        if headers[:3] == ['Task', 'Metric ↑', 'd20 / step 17,100 ↑']:
            return highlight_rows(rows, [2, 3])
        if headers[0] == 'Model / protocol':
            groups = [('corrected', row[1], row[2]) if '· corrected' in row[0] else ('other', i)
                      for i, row in enumerate(rows)]
            return highlight_columns(rows, {3: 'max', 4: 'max'}, groups)
        return rows

    return map_tables(text, transform)


def later_markdown(data, records):
    find = lambda sub: next(r for r in records if sub in r['path'])
    official = data['official_cetvel_receipt']
    texts = ['# Additional LLM configurations and recovered benchmarks\n',
        '**Direction legend:** ↑ higher is better for the named metric; ↓ lower is better; ↔ identifies configuration, budget or diagnostic values without a universal quality direction. Scores require the same task/protocol, and speed, elapsed time and memory require matched workload/hardware. Legacy scores remain provisional. Read the [metric guide](metric_guide.md) for definitions and limits.\n',
        '**Bold numeric table cells** mark the best displayed score across the models on the same legacy task row, or within the corrected XQuAD group sharing protocol, generation cap and sample count. Displayed ties are all bold; legacy rankings remain provisional. Different task/category scores, unmatched runtime workloads, budgets and training-overlap BPB receive no winner highlighting.\n',
        'These existing checkpoints and experiment outputs were recovered from UHeM on 9 November 2026. They extend the original [eleven-model tokenizer comparison](model_results.md) with continuation, model scaling, generation-protocol corrections and broader benchmarks. No new training or evaluation is claimed. Download [configuration CSV](model_configuration_results.csv), [complete JSON](model_configuration_results.json), [all available task metrics](model_all_task_metrics.csv), and [table-compatible protocol task scores](model_official_task_metrics.csv).\n',
        '## Model configuration and training results\n',
        '**The d32 BPB is a training-overlap diagnostic.** Its launch manifest makes all 30 source shards eligible for optimization and reuses one source shard for validation. It must not be presented as a held-out BPB improvement over d20 or MorphBPE. The completed FineWeb-2 d32 experiment is distinct from the separate general-corpus d32/WSD production plan.\n',
        'BPB ↓ is the prediction cost in bits per text byte. Layers, width, heads, parameters, context and token/step counts ↔ describe architecture and budget, not outcome quality; larger values are not automatically better.\n',
        table(['Experiment', 'Layers / width / heads ↔', 'Parameters ↔', 'Checkpoint ↔', 'Batch positions ↔', 'Cumulative positions ↔', 'BPB ↓', 'Training-loop h ↓'], [[r['experiment'], f"{r['depth']} / {r['width']} / {r['heads']}", f"{r['total_parameters']:,}" if r['total_parameters'] else '≈896.5M (baseline design)', f"{r['checkpoint_step']:,}", f"{r['global_batch_token_positions']:,}", f"{r['scheduled_cumulative_token_positions']:,}", number(r['validation_bpb'],8), number(r['training_loop_hours'],3)] for r in data['runs']]),
        '\nBoth use raw BPE with 32,768 entries, 2,048-token context and full attention. d20 continues the 17,100-step baseline to 34,200 cumulative steps with an explicitly changed continuation learning-rate schedule (start fraction 0.05, peak 0.25, 1,000-step warmup). Its BPB changes from archived 0.62321265 to 0.59145082; there is no recovered continuation CETVEL result. The checkpoint time counter is cumulative state and is not attributed solely to the continuation phase.\n',
        'd32 uses 2,818,575,450 parameters, BF16, eight A100 GPUs, SDPA and 58,720,256,000 scheduled positions. Its 28,000 steps are a coverage-oriented budget; the legacy loader/resume behavior does not establish exactly-once exposure. Training-loop hours exclude initial compile/warmup steps, evaluation, checkpointing and launch overhead. Exact metadata and the original launch manifest are preserved in JSON.\n',
        '### Measured performance and scheduler outcomes\n',
        '**The d20 continuation saved its final step-34,200 checkpoint, but Slurm job 512018 ended FAILED (exit code 1).** Its final model measurements are available; it is not described as a cleanly completed job. The d32 job 512605 completed with exit code 0.\n',
        'Throughput ↑ counts processed work per second; step time ↓ measures time per update; wall time ↓ measures the whole job; peak MiB ↓ measures maximum logged GPU-memory demand. MFU ↔ estimates the fraction of theoretical compute used and does not measure prediction quality. These efficiency directions apply only to matched workloads, units and hardware.\n',
        table(['Experiment','Job / state','Slurm wall time ↓','Median throughput ↑','Unit','Median step ms ↓','BF16 MFU % ↔','Peak MiB ↓'], [[r['experiment'],f"{r['slurm_job_id']} / {r['slurm_state']}",r['slurm_wall_clock'],number(r['median_last100_throughput'],1),r['throughput_units'],number(r['median_last100_dt_ms'],2),number(r['median_last100_bf16_mfu_percent'],2),number(r['peak_memory_mib'],2)] for r in data['runs']]),
        '\nThe medians use the final 100 printed iterations: d20 steps 34,100–34,199 and d32 updates 27,901–28,000. Both use eight A100s. The d20 logger measures token positions/s, whereas d32 measures valid prediction targets/s; preserve this distinction. The d20 job wall time is 12:11:58, while its inherited cumulative training-time counter is 33.203 hours. These durations cover different phases and must not be divided into the same token budget. Whole-log hashes and selected line numbers are retained in the evidence bundle.\n',
        '## Legacy core-11: d20 and d32\n',
        'The following preserves the original eleven classification/log-likelihood metrics. The legacy adapter did not validate the tokenized context/continuation boundary, so these values may be affected by likelihood-scoring errors. Their differences are provisional descriptive measurements, not established model-scale or tokenizer effects; a corrected common rerun is needed. The models also differ in scale, data eligibility and training budget despite sharing raw BPE.\n']
    old = find('full_kvcache_')['data']['results']; scaled = find('core11_legacy_')['data']['results']
    tasks = ['exams_tr','belebele_tr','turkish_plu','cetvel_xcopa_tr','cetvel_xnli_tr','mnli_tr','snli_tr','news_cat','offenseval_tr','trclaim19','xfact_tr']
    rows=[]; a=[]; b=[]
    for t in tasks:
        metric='acc,none' if t=='cetvel_xcopa_tr' else 'acc_norm,none'
        a.append(old[t][metric]); b.append(scaled[t][metric]); rows.append([f'`{t}`', metric.split(',')[0], number(a[-1]),number(b[-1]),f'{100*(b[-1]-a[-1]):+.2f}'])
    rows.append(['**Core-11 mean**','mean',number(mean(a)),number(mean(b)),f'{100*(mean(b)-mean(a)):+.2f}'])
    texts += ['Accuracy ↑ counts correct answers; `acc_norm` ↑ first selects answers by length-normalized likelihood. Core-11 ↑ averages the eleven task scores. Positive deltas mean a higher recorded score; this does not resolve the legacy scoring caveat.\n',
        table(['Task','Metric ↑','d20 / step 17,100 ↑','d32 / step 28,000 ↑','Δ percentage points ↑'],rows),
        '\n## Generation protocol: corrected XQuAD\n',
        'The legacy adapter misread generation kwargs, ignoring actual stop strings and task generation limits; its XQuAD values cannot support absolute QA or tokenizer-effect conclusions. The original scores are retained as historical evidence. Only the raw-BPE d20 reference and d32 have recovered corrected-generation runs in this bundle. They do not establish corrected scores for the other ten tokenizer-trained models.\n']
    rows=[]
    for label, sub, budget in [('d20 baseline · legacy','full_kvcache_',128),('d20 baseline · corrected','reference_d20_xquad_fixed',128),('d32 · corrected','/xquad_fixed_step',128),('d32 · Kumru comparison protocol','/kumru20_qa_fixed',256)]:
        rec=find(sub); vals=rec['data']['results']['xquad_tr']; rows.append([label,budget,1190,number(vals['exact,none']),number(vals['f1,none'])])
    texts += ['Exact match ↑ requires the whole normalized answer to match a reference. QA F1 ↑ balances precision and recall of overlapping answer words. Example counts and generation limits ↔ describe the evaluation setup, not quality.\n',
        table(['Model / protocol','Max generation tokens ↔','Examples ↔','Exact match % ↑','F1 / 100 ↑'],rows),
        '\nThe corrected 128-token d20/d32 comparison gives F1 11.3369 versus 15.8833. The later 256-token d32 QA run gives 15.1717; both are retained with their own source paths and settings. Generation limits and the recorded protocol must accompany any score. The same-checkpoint d20 change is an evaluator correction, not a training gain; the d20/d32 contrast also changes model scale and training exposure. See [the evaluator audit](later_model_results.md) for the recovered bug mechanisms.\n',
        '## Complete d32 CETVEL EACL table-compatible scoring\n',
        f"The saved receipt reports **{official['overall_score']:.4f}/100**, the unweighted mean of seven category scores. The verification receipt covers **{data['official_verification']['total_effective_samples']:,} effective samples**, 24 top-level tasks and 38 verified leaves, with no sample limit. It records all eight source-shard hashes. This is the repository's `cetvel-eacl2026-table4-compatible-v1` protocol, not the legacy core-11 macro or an independent verification against an official public leaderboard.\n",
        'Classification uses raw accuracy ↑, QA normalized exact match ↑, summarization TabiBERT BERTScore F1 ↑, translation COMET-22 ↑, and GEC stripped exact match ↑. BERTScore compares contextual word representations; COMET estimates translation quality with a learned evaluator. The category and overall scores are higher-is-better only within this fixed protocol, not interchangeable measures across tasks. Do not substitute legacy `acc_norm`, ROUGE, BLEU or QA F1 into this aggregate. The receipt documents table-compatibility assumptions, model revisions and metric settings.\n',
        table(['Category','Score / 100 ↑'], [[k,number(official['category_scores'][k])] for k in official['protocol']['category_order']] + [['**Overall**',f"{official['overall_score']:.4f}"]]),
        '\n### All table-compatible protocol task scores\n',
        table(['Task','Metric ↑','Examples ↔','Score / 100 ↑'],[[f'`{t}`',d['metric'],d.get('sample_count') if d.get('sample_count') is not None else 'aggregate',number(d['score'])] for t,d in official['task_details'].items()]),
        '\nThe MCQA definition includes both the Turkish-PLU aggregate and its listed component tasks, as specified by the archived table-compatible receipt. The translation task identifiers have counterintuitive direction names; use the receipt’s explicit source/target-language fields. The seven-category macro deliberately weights categories equally.\n',
        '## Other recovered d32 metrics\n',
        'The earlier Kumru-comparison output files also contain auxiliary metrics. These are separate measurements/aggregations from the table-compatible receipt above. Values retain their original scale. All auxiliary scores and available standard errors—including NLU subtasks—are included in the all-metric CSV and JSON. Standard errors describe uncertainty rather than model quality.\n',
        'BLEU ↑ measures matching word sequences in translation, ROUGE ↑ measures overlap with reference summaries, and WER ↓ measures word insertions, deletions and substitutions relative to reference length. QA F1 ↑ measures answer-word overlap; EM/exact/exact_match ↑ require a matching answer after the recorded normalization. The metric column marks the preferred direction for each row; compare only the same metric, task, normalization and protocol.\n']
    rows=[]
    specs=[('/kumru20_qa_fixed',{'mkqa_tr':['em,none','f1,none'],'tquad':['exact,none','f1,none'],'xquad_tr':['exact,none','f1,none']}),('/kumru20_gec_mt_fixed',{'gecturk_generation':['exact_match,regex_filter_applies_strip_internally'],'wmt-tr-en-prompt':['wer,none','bleu,none']}),('/kumru20_sum_fixed',{'mlsum_tr':['rouge1,none','rouge2,none','rougeL,none','rougeLsum,none'],'tr-wikihow-summ':['rouge1,none','rouge2,none','rougeL,none','rougeLsum,none']})]
    for sub,taskspec in specs:
        rec=find(sub); vals=rec['data']
        for t,metrics in taskspec.items():
            for m in metrics:
                scale='0–100' if t in {'tquad','xquad_tr'} else 'native output'
                metric_label = m.split(',')[0] + (' ↓' if m.startswith('wer,') else ' ↑')
                rows.append([f'`{t}`',metric_label,vals['n-samples'][t]['effective'],number(vals['results'][t][m],6),scale])
    texts += [table(['Task','Metric / direction','Examples ↔','Value','Scale'],rows),
        '\nThe `kumru20_sum_fixed` file is a partial suite result containing complete outputs for the two listed summarization tasks; it is not a completed full-suite evaluation. The later table-compatible receipt includes four summarization tasks under BERTScore.\n',
        '## Evidence coverage and unavailable results\n',
        table(['Evidence','Disposition'],[
            ['11 original tokenizer-trained models','Archived scores retained; 132 primary cells checked against recovered full-precision outputs.'],
            ['32k TurkishDelight full d20 model','No completed result in the archived/recovered inventory.'],
            ['d20 continuation step 34,200','Checkpoint metadata present; no continuation benchmark found in this bundle.'],
            ['d32 step 28,000','Checkpoint, launch manifest, full table-compatible receipt, corrected QA and legacy/auxiliary evaluations present.'],
            ['d32 smoke_job515374 / kumru20_comparison_smoke_job515380','Two examples per task; retained in JSON/CSV with is_smoke=true; excluded from capability tables.'],
            ['Raw d20 cetvel_out/full/cetvel_full_results.json','Recovery reports malformed JSON; not used. The valid archived partial/core comparison is retained.'],
            ['Original raw d20 meta_017100.json','Absent at recovery; archived checkpoint statistics retained.'],
            ['Later d32 SFT configuration screens','Reported in the separate [SFT screen report](sft_configuration_screens.md); these vary post-training settings on one raw-BPE model.'],
            ['General-corpus d32 WSD','Separate production-plan documents; not the completed FineWeb-2 d32 model reported here.'],
            ['June 7 d2 pipeline smoke','Separate archived infrastructure result; see model report.'],
        ]),
        '\nRebuild all tables with `python -m scripts.build_model_study_results`. The builder verifies recovered primary task values, full-precision macros, checkpoint identity and receipt category/overall arithmetic. Source artifacts retain retrieval timestamps, UHeM paths, original source-file hashes, settings and reported errors.\n']
    return '\n'.join(texts)

# Segmenter Examples

Small original-vs-segmented examples for the Turkish morphology tokenizer study.
These examples are committed for quick inspection only. Full corpus outputs are
too large for git and should stay under `dev-ignore/`.

## Sentence Examples

Original:

```text
Türkiye'de evlerden geldik.
Kitaplarımı masanın üstünden aldım.
Çalışıyorum ama anlaşamadık.
Öğrencilerimizden bazıları İstanbul'dan Ankara'ya gitti.
```

Identity baseline:

```text
Türkiye'de evlerden geldik.
Kitaplarımı masanın üstünden aldım.
Çalışıyorum ama anlaşamadık.
Öğrencilerimizden bazıları İstanbul'dan Ankara'ya gitti.
```

TRmorph:

```text
Türkiye'de ev ler den gel dik.
Kitaplarımı masa nın üstün den al dı m.
Çalış ıyor um ama anlaş a ma dık.
Öğrenci ler imiz den bazı ları İstanbul'dan Ankara'ya git ti.
```

Zemberek:

```text
Türkiye'de ev ler den gel di k.
Kitap lar ım ı masa nın üst ün den al dı m.
Çalış ıyor um ama anlaş ama dı k.
Öğrenci ler imiz den bazı lar ı İstanbul'dan Ankara'ya git ti.
```

TurkishDelightNLP:

```text
Unavailable in the current local environment. The adapter is implemented, but a
TDELIGHT_SEGMENT_CMD or TDELIGHT_URL backend has not been configured yet.
```

## Word Examples

| Word | TRmorph pieces | Zemberek pieces |
| --- | --- | --- |
| `evlerden` | `ev + ler + den` | `ev + ler + den` |
| `geldik` | `gel + dik` | `gel + di + k` |
| `kitaplarım` | `kitap + lar + ım` | `kitap + lar + ım` |
| `çalışıyorum` | `çalış + ıyor + um` | `çalış + ıyor + um` |

## Commands

TRmorph example command:

```bash
TRMORPH_SEGMENT_FST=/private/tmp/TRmorph/segment.fst \
TRMORPH_FLOOKUP_FLAGS=-x \
python3 -m scripts.morph_segment \
  --backend trmorph \
  --format text \
  --text "Türkiye'de evlerden geldik."
```

Zemberek example command:

```bash
ZEMBEREK_SEGMENT_CMD="/private/tmp/zemberek-smoke-py312/bin/python scripts/zemberek_segment_cmd.py" \
python3 -m scripts.morph_segment \
  --backend zemberek \
  --format text \
  --text "Türkiye'de evlerden geldik."
```

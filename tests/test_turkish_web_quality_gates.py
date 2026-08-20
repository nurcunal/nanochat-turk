from __future__ import annotations

from pathlib import Path

import pytest

from nanochat.turkish_corpus import audit_document, load_corpus_policy


POLICY = load_corpus_policy(
    Path("configs/pretrain/tr_d32_turkish_general_v2.json")
)["content_policy"]
STRICT_ENCODING_POLICY = {
    **POLICY,
    "max_unicode_replacement_characters": 0,
    "max_mojibake_sequence_hits": 0,
    "max_c1_control_characters": 0,
    "max_unicode_surrogate_characters": 0,
}

BASE = (
    "Bugün mahallede arkadaşlarımla buluştuk ve uzun zamandır konuşmadığımız "
    "konuları sakin biçimde değerlendirdik. Herkes kendi deneyimini anlattı, "
    "çünkü günlük yaşamda karşılaştığımız sorunlara birlikte çözüm bulmanın daha "
    "yararlı olduğunu düşünüyoruz. Sonra komşuların hazırladığı yemekleri paylaştık, "
    "çocukların okul çalışmalarını konuştuk ve yarın yapılacak etkinlik için görevleri "
    "belirledik. Böyle toplantılar insanları birbirine yaklaştırıyor, yeni bilgiler "
    "öğrenmemizi sağlıyor ve yaşadığımız çevreyi daha iyi anlamamıza yardımcı oluyor. "
)


def _decision(extra: str = "", *, url: str = "https://ornek.test/haber/gunluk-yasam"):
    return audit_document(
        BASE + extra,
        url=url,
        source_lid_ok=True,
        content_policy=POLICY,
    )


@pytest.mark.parametrize(
    ("corruption", "metric"),
    [
        (" bozuk\ufffdmetin", "unicode_replacement_characters"),
        (" gÃ¼zel görünen ama bozuk metin", "mojibake_sequence_hits"),
        (" görünmez\x85denetim işareti", "c1_control_characters"),
        (" eşlenmemiş\ud800kod noktası", "unicode_surrogate_characters"),
    ],
)
def test_strict_encoding_gate_rejects_corrupt_native_text(
    corruption: str, metric: str
) -> None:
    decision = audit_document(
        BASE + corruption,
        source_lid_ok=True,
        content_policy=STRICT_ENCODING_POLICY,
    )
    assert decision.reason == "text_encoding_corruption"
    assert decision.metrics[metric] >= 1


def test_strict_encoding_gate_keeps_valid_turkish_circumflexes() -> None:
    decision = audit_document(
        BASE + " Kâğıt üzerindeki resmî açıklama, millî kültür ve sükûnetten söz ediyor.",
        source_lid_ok=True,
        content_policy=STRICT_ENCODING_POLICY,
    )
    assert decision.accepted
    assert decision.metrics["mojibake_sequence_hits"] == 0


def test_foreign_script_gate_requires_both_absolute_and_fractional_thresholds():
    long_cyrillic = " Русский текст показывает другую страницу и повторяется снова здесь."
    rejected = _decision(long_cyrillic)
    assert rejected.reason == "foreign_script"
    assert rejected.metrics["foreign_script_characters"] >= 32
    assert rejected.metrics["foreign_script_fraction"] > 0.02

    short_native_name = _decision(" Rusça adı: Южно-Сахалинск olarak da yazılır.")
    assert short_native_name.accepted
    assert short_native_name.metrics["foreign_script_characters"] < 32


@pytest.mark.parametrize(
    ("extra", "url"),
    [
        (
            " Bu sürüm için Mod APK indir bağlantısını kullanın, ardından ücretsiz indir düğmesine basın.",
            "https://ornek.test/telefon/oyun",
        ),
        (
            " Telefon hakkında sıradan bir değerlendirme yazısıdır.",
            "https://tr.apkpure.com/sade-bir-sayfa",
        ),
    ],
)
def test_contextual_software_download_pages_are_rejected(extra: str, url: str):
    assert _decision(extra, url=url).reason == "software_download"


def test_generic_software_words_and_one_download_phrase_remain_accepted():
    decision = _decision(
        " Telefon uygulama güncellemesi kullanıcıların güvenliğini iyileştiriyor; "
        "kitabın örnek bölümünü ücretsiz indir seçeneği de haberde açıklanıyor.",
        url="https://ornek.test/haber/telefon-guncellemesi",
    )
    assert decision.accepted
    assert decision.metrics["software_download_hits"] == 1


def test_commerce_requires_two_signals_or_one_signal_plus_commerce_url():
    assert _decision(
        " Ürün 24 saatte kargoda, stokta ve taksit seçenekleri ile sunuluyor."
    ).reason == "commerce"
    assert _decision(
        " Yeni kitap stokta göründüğü için okurlar sevindi.",
        url="https://ornek.test/kitap/yeni-roman",
    ).reason == "commerce"
    assert _decision(
        " Haberde temel ürünün artık stokta olduğu açıklandı.",
        url="https://ornek.test/haber/tedarik",
    ).accepted


def test_cookie_legal_taxonomy_and_seo_templates_are_narrowly_rejected():
    assert _decision(
        " Çerez ayarları bölümünü açın ve tüm çerezleri kabul et düğmesine basın."
    ).reason == "cookie_ui"
    assert audit_document(
        "KVKK Aydınlatma Metni\n" + BASE,
        url="https://ornek.test/kvkk",
        source_lid_ok=True,
        content_policy=POLICY,
    ).reason == "legal_policy"
    assert _decision(
        " KVKK konusunda yapılan yeni düzenleme uzmanlar tarafından tartışıldı.",
        url="https://ornek.test/haber/kvkk-degisikligi",
    ).accepted
    assert _decision(url="https://ornek.test/kategori/gundem").reason == "taxonomy_url"
    assert audit_document(
        "Bu sayfada dayanışma nedir ve toplum için ne demek sorularını açıklıyoruz. "
        + BASE,
        url="https://ornek.test/dayanisma",
        source_lid_ok=True,
        content_policy=POLICY,
    ).reason == "seo_definition_template"

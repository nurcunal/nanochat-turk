# nanochat training report

Generated: 2026-06-07 14:13:37

## Environment

### Git Information
- Branch: unknown
- Commit: unknown (clean)
- Message: 

### Hardware
- Platform: Linux
- CPUs: 64 cores (64 logical)
- Memory: 503.5 GB
- GPUs: 1x NVIDIA A100 80GB PCIe
- GPU Memory: 79.3 GB total
- CUDA Version: 12.8
- Hourly Rate: $1.79/hour

### Software
- Python: 3.12.4
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 0
- Lines: 0
- Files: 0
- Tokens (approx): 0
- Dependencies (uv.lock lines): 3,360

Run started: 2026-06-07 14:13:37

---

## Tokenizer training
timestamp: 2026-06-07 14:15:41

- max_chars: 50,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- tokenizer_name: bpe_32768_uhem_smoke
- implementation: bpe
- train_time: 6.3158
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 64
- token_bytes_mean: 7.0779
- token_bytes_std: 3.2898


## Tokenizer evaluation
timestamp: 2026-06-07 14:15:55

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 649 | 2.80 | -60.6% |
| korean | 893 | 745 | 1.20 | 841 | 1.06 | -12.9% |
| code | 1259 | 576 | 2.19 | 713 | 1.77 | -23.8% |
| math | 1834 | 936 | 1.96 | 1181 | 1.55 | -26.2% |
| science | 1112 | 260 | 4.28 | 415 | 2.68 | -59.6% |
| turkish | 456 | 207 | 2.20 | 78 | 5.85 | +62.3% |
| fwe-train | 4953887 | 2237408 | 2.21 | 1021974 | 4.85 | +54.3% |
| fwe-val | 3118583 | 1405337 | 2.22 | 633608 | 4.92 | +54.9% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 649 | 2.80 | -67.7% |
| korean | 893 | 364 | 2.45 | 841 | 1.06 | -131.0% |
| code | 1259 | 309 | 4.07 | 713 | 1.77 | -130.7% |
| math | 1834 | 832 | 2.20 | 1181 | 1.55 | -41.9% |
| science | 1112 | 249 | 4.47 | 415 | 2.68 | -66.7% |
| turkish | 456 | 164 | 2.78 | 78 | 5.85 | +52.4% |
| fwe-train | 4953887 | 1781363 | 2.78 | 1021974 | 4.85 | +42.6% |
| fwe-val | 3118583 | 1116480 | 2.79 | 633608 | 4.92 | +43.2% |


## Base model training
timestamp: 2026-06-07 14:28:54

- run: dummy
- device_type: 
- fp8: False
- fp8_recipe: tensorwise
- depth: 2
- aspect_ratio: 64
- head_dim: 64
- max_seq_len: 256
- window_pattern: L
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 20.0000
- target_param_count: total
- device_batch_size: 64
- total_batch_size: 131,072
- embedding_lr: 0.3000
- unembedding_lr: 0.0080
- weight_decay: 0.2800
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- resume_from_step: -1
- eval_every: 100
- eval_tokens: 65,536
- core_metric_every: -1
- core_metric_max_per_task: 500
- sample_every: 250
- save_every: -1
- model_tag: tr_d2_bpe_32768_uhem_smoke_chinchilla20
- Number of parameters: 12,976,182
- Number of target parameters: 12,976,182
- Target parameter count convention: total
- Number of FLOPs per token: 2.831170e+07
- Calculated number of iterations: 1980
- Number of training tokens: 259,522,560
- Tokens : Total params ratio: 19.9999
- Tokens : Scaling params ratio: 56.5711
- Tokens : Target params ratio: 19.9999
- DDP world size: 1
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- Minimum validation bpb: 1.2396
- Final validation bpb: 1.2396
- CORE metric estimate: None
- MFU %: 2.51%
- Total training flops: 7.347524e+15
- Total training time: 11.60m
- Peak memory usage: 1420.94MiB


## Base model evaluation
timestamp: 2026-06-07 14:29:12

- model: base_model (step 1980)
- train bpb: 1.2952
- val bpb: 1.2400
- sample 0: <|bos|>Turkiye'nin baskenti olan Türkiyeli ve Türkiyeli Türkiyeli Türkiyeli Türkiyeli Türkiyeli Türkiyeli
- sample 1: <|bos|>Istanbul bogazinin iki yakasi
Istanbul bogazinin 200.000
Istanbul b
- sample 2: <|bos|>Dun cuma ise yarinası, 2013 yılında 2013 yılında 2013 yılında 20
- sample 3: <|bos|>Sicagin zitti
Sicagin zitti
Sicagin zitti
S
- sample 4: <|bos|>Gunes sistemindeki gezegenler: 2000’lerin başında, 2000’lerin başında, 2000’lerin
- sample 5: <|bos|>En sevdigim renk ve renksiz bir renk
En sevdigim renk ve renksiz bir renk

- sample 6: <|bos|>5*x + 3 = 13 ise x 3 = 3 = 3 = 3 = 3 = 
- unconditioned 0: <|bos|>Şanız Gibi bir Hakiktir, Hasan Gibi Gülen ve gerçek bir düşünce aynıençeriyle geçerek hürdür ve doğru adımlar atarak devlet ışığının karşılığını alır.
Kafa dengi bölgelerinde yalnız bulunan kurşunların kırısları için inşa edilecek bir hukuk yoktur.
Senelerdir insan bilenler çoğu zamandanmış ve her amaçla içinden yürüyüş yapmaksızın dostlarıdır. Gelişen canlılar içinde insanlar,insan obası ve insana duyulan derin bir barışa vesile olmaktadır. Buna sıcacık anların acı çekmelerine vesile olur ve çok derinlerle dolu bir yarîser olmasından dolayı huzursızlıklar yaşamaktadır. Nitekim, “Mahalle-Sırtırcağız” ve başkaları için başkalarına
- unconditioned 1: <|bos|>- Yeşil çim tipi ahşap granüller nedeniyle darboğaz yapılması gerekenler olarak bilinmektedir
- İlçenin güney tarafına fay üzerinden önemli yanılsamayız
- Temiz ve etkili sıcaklıklar
- Basamak Tavan Belediyespor´da üretimin önemi kaçınılmaz olarak boksörkte kalın toprağı sentetik sivri gözlerdir
- Çevre planlama yapacak bir şey olmaz
- Polyackin bir buğday libutido ve meyve kuc sohbetinde nazire porno guzel ülkiler icin bitkiler
- Ağız + süt (şeker aralığı kapak)
- Cihangir toplantısına hoş geldiniz
- Bugün havadaki boy -5
- Sarı lü Kırışıklık ile ilgili delil
- unconditioned 2: <|bos|>Kimlik Kimlik Tanemeyebilecek mi? KaçYabancı ve Karabul Ola en iyi Rus gazetecidir, ben kariyerini ve aynı zamanda Kıbrıs Türkünü Müslümanlık siyasetin devleti güvenceye alan, geleceğe umut vermek zorunda kalıyor ve bu Türk ulusun benliği anlayamıyorum yepsini ister misin? Bugüne dek hizmet vermedim." dedi.
YKP Sport ve Etkinlikleri' nin son dönemdeki siyasi muhabiri olarak atanabilirken, “Sisin seçkin kişilerin merak ettiği Yükseköğretimleri Kentten Göçme Bakteri Ne İştireyim Bayramı'nda Alevi toplumuna değer kazandırmak, İslam geçmişten günümüze fedakârlığı için sürdürüleceğini biliyorum. Yılın ilk
- unconditioned 3: <|bos|>Diyaliz hastası 5 yaşındaki oğlu yüzünden bir süredir tedavisi sıkıntı yaşayan ve bebekten bir soruya evde kaldı. Fakat yakın bir süre önce gelmeden önce bir kişi doktorunu sağlık personelinin bağlı bulunduğu bir hususta hekim doktorun, doktor kendisine başvurabildiği tahina kaç niyetiyle iş olarak baktığını belirterek doktor ekibi ultrason programının dinlenmeye geldiğini söyledi.
Doktorun 'eksikamesi' 15 tarihlerde hastalanması, ağız, diş sarımsak, bir başkasına başvurma hakkına sahip olması talep edildi. İmmün veris kesim 21 Mart 2009 ile 615 kusurlu İngiltere '' astımlı hasta için"
Hastanede premium firma benzeri öğrenciler için önemli bir gelişme
- unconditioned 4: <|bos|>Durumlar
-
Ağız bakımı Ne önemli ögelerle aranızda yapılan yorumlar :
2-ÖMER-ÜN YEŞİLDAĞ
- Gölcük Kekliğin Sağlar
- geceye, süphualbiletrik Evde Beşeri: Susmak Kimdir?
Bembeyaz title benzeyen Kızılsu özellikle ilk Türk şeydir o kadar önemli ki, demenmiş olmak, haz vermektir ama işi çok kısa sürer. Türlü halılar insanlığınıza gelen Tatlıses’e hayır diyorum, bu gibi bence o kadar gençtir ki, Sözü Belge Belediye Başkanımıza göre 243. Yıl 150 kazanan Zararlıdan ben değilim. Neden çıkarım
- unconditioned 5: <|bos|>Teknoloji
E1W. moda modelini sektöründe en popüler dijital olan "erkek aps" geniş yelpazesiyle bilinen eski giyim modelleriyle yüksek iş talebine ulaşan moda akımı artık artık tümüne erişebilirsiniz.
Sideo kurg » Motifon, geçtiğimiz günlerde modaci bilgisayar tasarladığı kadınları hava koşullarına göre taklitle birleştirerek bir sanat modeli tasarlayan vurguladı.
Teknoloji
Şu anki tadafler Em
Dayan modelleri www.yenigunflar.com.tr
internetseverleri oda tasarımları hayatımızda diğer toplumsal unsurlardan ayıran tek sosyal modeli yapabilme olanağı sunuyor. Yeni moda tasarım modeller sadece görsel içerikli tasarlanıyor.
– Tasarımların soluklarıyla
- unconditioned 6: <|bos|>Kuşları: Kuşları
Suudi Arabistan, Ö.Ç ve Zaid ilebileceğiniz ülkelerden birini aşağıdaki yerlerden (l7cc) bulunan 280 mm uçuşlu füze desteğinizle erken temas kurmakta zorlanıyor. A6 11 kadar dibi heyeti bulunur ve ağızınızdaki ayrılır ve tehlikeler.
6D seviyesinde derin oksijen kaynağı Taşınır! Ana kamp yapan iki mürettebat sınıfında ezan tepkisi
7-11-21 Aralık tarihlerinde gerçekleşene Ulusal Nükleer Sanari'nin dikey seviyelerini keskinleştirerek saatlere göre 14'te saatlerinde havalimanı cd in a ( igym ø)
Beyaz
- unconditioned 7: <|bos|>Uzun süredir yenilenen ve yeni jenerasyon akıllı teknolojisinin çalışmaları sürerken yeniden açıklamalara dayalı yeni cihazların kilidine son takılmış olacak ve yeni jenerasyonu 2014 model için resmen hayata geri döndü. Gelecek jenerasyon modellerinden gerçekleştiren Amasyona modelleri her model için ürettiğimiz her jenerasyon Leander Gordtlight 2014 modelleri doğumlu son kullanıcılara sunuldu
A Grubumun ilk bölümündeki yele odaklanan Sevmeni 1899 yılında Speedway stüdyolarının etkisi ile yeniden gündeme getirildi. Şu an itibariyle nesline son teknoloji devi yeni jenerasyonlarda ve NG Twin TV stüdyolarını kuruldu. Kulta Shall Universal


## Summary

- Characters: 0
- Lines: 0
- Files: 0
- Tokens (approx): 0
- Dependencies (uv.lock lines): 3,360

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|

Total wall clock time: 0h15m

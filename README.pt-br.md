# facetrust — métricas de biometria facial sob ataques de morphing e watermarking

Base de código para o demonstrador do tutorial *"Biometrics under attack: On provenance, trust and threat robustness"*. Implementa as métricas de **identificação/verificação biométrica** e de **qualidade de imagem** descritas no artigo mencionado acima e nos slides, para avaliar ataques (morphing) e defesas (watermarking — SSLWatermarking, RoSTEALS) ao longo do pipeline:

```
captura -> processamento -> registro (enrollment) -> reconhecimento (decisão)
```

**O que este pacote NÃO faz:** não gera watermarks, não decodifica mensagens de SSLWatermarking/RoSTEALS, e não gera morphings. Ele assume que essas imagens (limpas, marcadas e morphed) já existem, e foca em transformar isso em métricas comparáveis, tabelas e gráficos.

## Instalação

```bash
pip install -e .                       # núcleo: verification/identification/
                                       # morphing/image_quality/pipeline/plotting
pip install -e ".[arcface]"            # + ArcFaceEmbedder pronto (insightface)
pip install -e ".[lpips]"              # + métrica perceptual LPIPS
pip install -e ".[arcface,lpips]"      # combine quantas extras quiser
pip install -e ".[all]"                # tudo, incluindo ssl-watermarking e rosteals
```

Sem nenhuma extra, o pacote já funciona normalmente: basta fornecer seus próprios embeddings via `CallableEmbedder`, e usar apenas PSNR/SSIM em vez de LPIPS.

`requirements.txt` continua disponível como alternativa para quem prefere `pip install -r requirements.txt` a instalar o pacote propriamente dito.

### Sobre as extras `ssl-watermarking` e `rosteals`

Nem o SSLWatermarking (`facebookresearch/ssl_watermarking`) nem o RoSTEALS (`TuBui/RoSteALS`) são pacotes publicados no PyPI — os dois são repositórios de pesquisa distribuídos como "clone o repo + `pip install -r requirements.txt` + baixe os pesos manualmente", sem `setup.py`/ `pyproject.toml` próprio. Ou seja, **`pip install -e ".[rosteals]"` nunca vai instalar o RoSTEALS em si** — essas duas extras só pré-instalam as dependências pesadas de runtime que os dois repositórios exigem (`torch` etc.), para o ambiente já estar pronto para:

1. clonar cada repositório manualmente (`git clone ...`) e baixar os pesos conforme as instruções de cada um;
2. apontar as funções `embed`/`extract` de cada repo para `facetrust.watermark.SSLWatermarkAdapter` / `RoSTEALSAdapter`.

As versões fixadas nessas duas extras são um ponto de partida best-effort — confira o `requirements.txt` oficial de cada repositório antes de fixar versões em produção, pois eles evoluem independentemente deste pacote.

## Estrutura

```
facetrust/
  embeddings.py     - extração de embeddings faciais (ArcFace ou custom)
  image_quality.py  - PSNR, SSIM, LPIPS (opcional)
  verification.py   - similaridade, genuine/impostor, FAR/FRR/ROC/DET/EER/TAR@FAR, embedding drift
  identification.py - CMC e rank-N accuracy (cenário 1:N)
  morphing.py       - MMPMR / FMMPMR / ProdAvg-MMPMR / RMMR (potencial de ataque de morphing; MMPMR, ProdAvg e RMMR conferidos contra a implementação de referência de Scherhag et al., https://github.com/dasec/mvr)
  watermark.py      - BER, taxa de detecção, adaptadores para SSLWatermarking e RoSTEALS
  transforms.py     - JPEG, resize, blur, crop, rotação (robustez)
  plotting.py       - distribuições, ROC/DET, varredura de alpha, curva MMPMR
  pipeline.py        - orquestra tudo acima em cima de um dataset de amostras

  adapters/
    ssl_watermarking.py - integração real com facebookresearch/ssl_watermarking
                          (chama o encode/decode deles, não reimplementa)

examples/
  alpha_sweep_demo.py       - equivalente ao "starter loop" dos slides
  demo_full_pipeline.py     - demo completo lendo uma pasta de dados
  ssl_watermarking_demo.py  - roda o SSLWatermarking de verdade sobre data/clean/

tests/
  test_metrics_synthetic.py            - testes de sanidade com dados sintéticos
  test_ssl_watermarking_adapter_glue.py - testes da camada de cola do adaptador SSL (sem torch)
```

## Uso rápido

```python
from facetrust.embeddings import CallableEmbedder
from facetrust.pipeline import Sample, EvaluationDataset, PipelineEvaluator

# Troque por ArcFaceEmbedder() ou seu próprio extrator.
embedder = CallableEmbedder(minha_funcao_de_embedding)

dataset = EvaluationDataset()
dataset.add(Sample("alice_clean", imagem_limpa, subject_id="alice", condition="clean"))
dataset.add(Sample("alice_wm_0.5", imagem_marcada, subject_id="alice",
                    condition="watermarked", alpha=0.5))
dataset.add(Sample("morph_ab_stylegan2", imagem_morphed, subject_id="alice+bob",
                    condition="morphed", contributors=["alice", "bob"],
                    morph_algorithm="stylegan2"))

dataset.compute_embeddings(embedder)
evaluator = PipelineEvaluator(dataset)

print(evaluator.verification_report("clean"))       # EER, TAR@FAR
print(evaluator.image_quality_matrix())               # PSNR/SSIM clean vs. outras condições
print(evaluator.embedding_drift_report("watermarked")) # drift por alpha
print(evaluator.morph_attack_report())                 # MMPMR/FMMPMR/RMMR por FAR alvo
print(evaluator.morph_attack_report(group_by_algorithm=True))  # idem, quebrado por morph_algorithm
print(evaluator.robustness_table(embedder))            # tabela JPEG/resize/blur/crop/rotação
```

Para o fluxo completo lendo imagens de disco, veja `examples/demo_full_pipeline.py` (estrutura de pastas esperada documentada no cabeçalho do arquivo) e `examples/alpha_sweep_demo.py` para o exemplo mínimo de varredura de força do watermark.

## Conectando SSLWatermarking e RoSTEALS

`facetrust.watermark` não reimplementa esses métodos — ele padroniza os resultados que eles já produzem. Para o **SSLWatermarking**, já existe uma integração pronta em `facetrust.adapters.ssl_watermarking` (chama o `encode.watermark_0bit`/`decode.decode_0bit` reais do repositório clonado, não reimplementa a matemática de embedding):

```python
from facetrust.adapters.ssl_watermarking import load_ssl_watermarking, make_single_image_adapter_fns
from facetrust.watermark import SSLWatermarkAdapter

ctx = load_ssl_watermarking(
    repo_path="/caminho/para/ssl_watermarking",   # git clone facebookresearch/ssl_watermarking
    model_path="/caminho/para/dino_r50_plus.pth", # backbone pré-treinado (ver README deles)
    normlayer_path="/caminho/para/out2048_yfcc_orig.pth",
    target_fpr=1e-6,
)
embed_fn, extract_fn = make_single_image_adapter_fns(ctx, mode="0bit")
ssl_wm = SSLWatermarkAdapter(embed_fn, extract_fn)
```

Requer `pip install -e ".[ssl-watermarking]"` (torch/torchvision/augly) + as dependências específicas do repositório deles + os pesos pré-treinados baixados manualmente (ver `facetrust/adapters/ssl_watermarking.py` para o passo a passo completo, incluindo um link direto para os pesos). Licença do ssl_watermarking: **CC-BY-NC** (uso não-comercial). Para processar um dataset inteiro de uma vez (mais eficiente que `SSLWatermarkAdapter` imagem por imagem), use `embed_0bit`/`decode_0bit` diretamente — ver `examples/ssl_watermarking_demo.py`. **Aviso de desempenho:** cada imagem passa por uma otimização iterativa (não um forward pass único) — rode isso offline, antes da sessão ao vivo do tutorial, não durante o hands-on.

Para o **RoSTEALS**, ainda não há uma integração pronta (a mesma estrutura de `ssl_watermarking.py` serve de modelo caso queira montar uma):

```python
from facetrust.watermark import SSLWatermarkAdapter, RoSTEALSAdapter

rosteals = RoSTEALSAdapter(embed_fn=meu_embed_rosteals, extract_fn=meu_decode_rosteals)
```

A partir daí, `bit_error_rate()` (RoSTEALS, baseado em payload) e `detection_metrics()` (SSLWatermarking, baseado em detecção) padronizam os resultados para entrarem em `PipelineEvaluator.robustness_table(..., watermark_detector=...)`.

## Comparando vários algoritmos de morphing

Uma morph não tem um único "sujeito" de referência (ela mistura dois ou mais contribuidores), então `Sample.subject_id` para `condition="morphed"` é só um rótulo de conveniência (sugestão: `"+".join(contributors)`, ex. `"alice+bob"`) — **não** é usado para achar uma referência "clean" única, porque essa referência não existe.

Quem identifica QUAL algoritmo gerou a morph é `Sample.morph_algorithm` (ex. `"stylegan2"`, `"stylegan3"`, `"diffae"`, `"opencv"`). Isso é o que permite avaliar vários algoritmos no MESMO par de contribuidores lado a lado:

```python
dataset.add(Sample("m1", img_stylegan2, subject_id="alice+bob",
                    condition="morphed", contributors=["alice", "bob"],
                    morph_algorithm="stylegan2"))
dataset.add(Sample("m2", img_stylegan3, subject_id="alice+bob",
                    condition="morphed", contributors=["alice", "bob"],
                    morph_algorithm="stylegan3"))
dataset.add(Sample("m3", img_diffae, subject_id="alice+bob",
                    condition="morphed", contributors=["alice", "bob"],
                    morph_algorithm="diffae"))

# Qualidade de imagem por algoritmo (1 linha por morph x contribuidor):
iq = evaluator.image_quality_matrix()
iq[iq["condition"] == "morphed"].groupby("morph_algorithm")[["psnr", "ssim"]].mean()

# MMPMR/FMMPMR/RMMR por algoritmo, na mesma varredura de FARs:
evaluator.morph_attack_report(group_by_algorithm=True)
```

Ao carregar de disco com `examples/demo_full_pipeline.py`, o algoritmo é a subpasta em `data/morph/<algorithm>/<contributor_a>+<contributor_b>.png` (ver cabeçalho do script para a convenção completa, incluindo o sufixo opcional para múltiplas morphs do mesmo par pelo mesmo algoritmo).

## Mapeamento artigo/slides → código

| Conceito no artigo/slides                              | Onde está no código                                   |
|----------------------------------------------------------|--------------------------------------------------------|
| Cosine similarity, genuine/impostor                       | `verification.cosine_similarity`, `build_score_pairs`  |
| FAR, FRR, threshold                                       | `verification.far_frr`, `threshold_at_far`             |
| ROC, DET, EER, TAR@FAR                                     | `verification.roc_curve_data`, `det_curve_data`, `eer`, `tar_at_far` |
| Embedding drift (`1 - cos(f(I), f(I_w))`)                  | `verification.embedding_drift`                          |
| Four core comparison scenarios (Clean/WM × Clean/WM)       | `verification.cross_condition_scores`, `pipeline.PipelineEvaluator.four_scenario_report` |
| CMC / rank-N (cenário de identificação)                    | `identification.cmc_curve`, `rank_n_accuracy`           |
| MMPMR / FMMPMR / ProdAvg-MMPMR / RMMR                       | `morphing.mmpmr`, `fmmpmr`, `prodavg_mmpmr`, `rmmr`, `mmpmr_at_far`, `map_curve` |
| PSNR, SSIM, (LPIPS)                                        | `image_quality.psnr`, `ssim`, `lpips_distance`          |
| Detection rate, BER, payload                               | `watermark.detection_metrics`, `bit_error_rate`         |
| Robustez a JPEG/resize/blur/crop/rotação                   | `transforms.DEFAULT_TRANSFORMS`, `pipeline.PipelineEvaluator.robustness_table` |
| Joint evaluation matrix                                    | `pipeline.PipelineEvaluator.joint_evaluation_summary`   |

## Testes

```bash
PYTHONPATH=. python -m pytest tests/ -v
# ou, sem pytest instalado, rode as funções test_* diretamente:
PYTHONPATH=. python -c "import tests.test_metrics_synthetic as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]; print('OK')"
```

Os testes usam apenas dados sintéticos (sem modelos de face nem watermarking reais), servindo como verificação de que a lógica das métricas está correta antes de rodar em cima dos seus dados reais.

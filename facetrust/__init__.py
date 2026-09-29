"""
facetrust
=========

Base de código para o tutorial "Biometrics under attack: On provenance,
trust and threat robustness".

Este pacote fornece as métricas discutidas no artigo/slides para avaliar,
de forma conjunta, o efeito de ataques (morphing) e defesas (watermarking,
ex. SSLWatermarking / RoSTEALS) em cada estágio do pipeline biométrico:

    captura -> processamento -> enrollment -> reconhecimento

Módulos:
    embeddings   - extração/normalização de embeddings faciais (ArcFace ou
                   qualquer extrator fornecido pelo usuário)
    image_quality- PSNR, SSIM e (opcional) LPIPS entre imagem original e
                   imagem atacada/marcada
    verification - similaridade de cosseno, distribuições genuine/impostor,
                   FAR/FRR, ROC/DET, EER, TAR@FAR, embedding drift
    identification - CMC e rank-N accuracy (cenário de identificação)
    morphing     - MMPMR, FMMPMR e RMMR (potencial de ataque de morphing)
    watermark    - BER, taxa de detecção e adaptadores para os métodos de
                   watermarking usados no tutorial (SSLWatermarking, RoSTEALS)
    transforms   - transformações de robustez (JPEG, resize, blur, crop,
                   rotação) aplicadas antes de recomputar as métricas acima
    pipeline     - orquestra os módulos acima ao longo dos estágios do
                   pipeline e produz as tabelas/figuras usadas no tutorial
    plotting     - funções de plot (distribuições, ROC/DET, curvas de
                   varredura de alpha, curva de MMPMR)

Nenhum destes módulos reimplementa um extrator de face ou um método de
watermarking: eles consomem embeddings e imagens já geradas pelo usuário
(conforme combinado, as imagens limpas, marcadas e "morphed" já existem).
"""

__version__ = "0.1.0"

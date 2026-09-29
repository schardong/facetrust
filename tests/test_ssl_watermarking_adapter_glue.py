"""
Testes da PARTE DO ADAPTADOR ESCRITA POR NÓS (não do ssl_watermarking em
si, que é importado e chamado como está -- não reimplementado). Cobrem
`_InMemoryImageDataset`, `_default_embed_params` e
`make_single_image_adapter_fns`, que são pura lógica de encanamento e não
precisam de torch/GPU/pesos para serem exercitados.

O que ISSO NÃO cobre (e não pode, neste ambiente): `load_ssl_watermarking`,
`embed_0bit`, `decode_0bit`, `embed_multibit`, `decode_multibit` -- todas
chamam código real do repositório ssl_watermarking (torch-dependente).
Precisam ser validadas rodando de verdade, com o repositório clonado e os
pesos baixados.
"""

from facetrust.adapters.ssl_watermarking import (
    _InMemoryImageDataset,
    _default_embed_params,
    make_single_image_adapter_fns,
)


def test_in_memory_image_dataset_applies_transform_and_dummy_label():
    images = ["img_a", "img_b", "img_c"]  # placeholders -- não precisam ser PIL de verdade aqui
    ds = _InMemoryImageDataset(images, transform=lambda img: img.upper())

    assert len(ds) == 3
    for i, expected in enumerate(["IMG_A", "IMG_B", "IMG_C"]):
        transformed, label = ds[i]
        assert transformed == expected
        assert label == 0  # rótulo fictício exigido pelo formato (imagem, rótulo) do dataloader deles


def test_default_embed_params_has_expected_fields():
    params = _default_embed_params(
        epochs=50, lambda_w=2.0, lambda_i=0.5, optimizer="Adam,lr=0.01", scheduler=None, verbose=1
    )
    assert params.epochs == 50
    assert params.lambda_w == 2.0
    assert params.lambda_i == 0.5
    assert params.optimizer == "Adam,lr=0.01"
    assert params.scheduler is None
    assert params.verbose == 1


def test_make_single_image_adapter_fns_rejects_invalid_mode():
    try:
        make_single_image_adapter_fns(ctx=None, mode="7bit")
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass


def test_make_single_image_adapter_fns_multibit_requires_payload():
    _, extract_fn = make_single_image_adapter_fns(ctx=None, mode="multibit")
    embed_fn, _ = make_single_image_adapter_fns(ctx=None, mode="multibit")
    try:
        embed_fn(image="qualquer", payload=None)
        assert False, "deveria ter levantado ValueError sem payload"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK   {name}")
    print("Todos os testes de cola passaram (sem exercitar torch/ssl_watermarking real).")

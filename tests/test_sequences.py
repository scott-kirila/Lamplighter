"""Sequence modelling: causal attention, next-token windows, and the language
-model recipe. The load-bearing property is that a causal model cannot see the
token it's asked to predict — everything else here supports that."""
import pytest
import torch

from lamplighter.backend.codegen import generate_dataloader, generate_module
from lamplighter.backend.diagnose import diagnose
from lamplighter.backend.inference import infer_shapes
from lamplighter.backend.recipes import RECIPES
from lamplighter.backend.runner import RunManager
from lamplighter.backend.schema import DataNode, Graph, ModelDef, ModelLink, Project
from tests.helpers import edge, graph, node

JOIN = 600


def _lm_graph(vocab=20, block=16, causal=True, embed=32, head_out=None, attention="TransformerEncoderLayer"):
    attn_params = (
        {"nhead": 4, "dim_feedforward": 64, "dropout": 0.0, "is_causal": causal}
        if attention == "TransformerEncoderLayer"
        else {"num_heads": 4, "dropout": 0.0, "is_causal": causal}
    )
    return graph(
        [
            node("in", "Input", {"shape": f"1, {block}", "dtype": "long"}),
            node("emb", "Embedding", {"num_embeddings": vocab, "embedding_dim": embed}),
            node("blk", attention, attn_params),
            node("head", "Linear", {"out_features": head_out or vocab}),
            node("out", "Output"),
        ],
        [edge("in", "emb"), edge("emb", "blk"), edge("blk", "head"), edge("head", "out")],
    )


def _lm_project(tokens_cfg=None, causal=True, vocab=20, block=16, **training):
    g = _lm_graph(vocab=vocab, block=block, causal=causal)
    return Project(
        models=[ModelDef(id="model", name="LM", graph=Graph(nodes=g.nodes, edges=g.edges))],
        data_nodes=[DataNode(id="d", kind="dataset", config={
            "source": "sequence", "corpus_var": "tokens", "block_size": block,
            "batch_size": 32, **(tokens_cfg or {})})],
        links=[ModelLink(id="L", source_data="d", target_model="model")],
        training={"recipe": "causal_lm", "epochs": 2, "device": "cpu", "lr": 3e-3,
                  "seed": 0, **training},
    )


# --- causal attention ---------------------------------------------------------

def test_causal_is_off_by_default_and_emits_nothing():
    src = generate_module(_lm_graph(causal=False))
    assert "is_causal" not in src and "generate_square_subsequent_mask" not in src


def test_causal_emits_a_real_mask_because_the_hint_alone_is_refused():
    # torch raises "Need attn_mask if specifying the is_causal hint" — so the
    # mask does the work and the flag only says it's triangular.
    src = generate_module(_lm_graph(causal=True))
    # The mask is sized from the batch's OWN sequence length at call time, and
    # built on the input's device.
    assert "src_mask=nn.Transformer.generate_square_subsequent_mask(t0.size(1), device=t0.device)" in src
    assert "is_causal=True)" in src
    # Self-attention takes the same mask under a different keyword.
    mha = generate_module(_lm_graph(causal=True, attention="MultiheadAttention"))
    assert "attn_mask=nn.Transformer.generate_square_subsequent_mask(" in mha


def test_a_causal_model_cannot_see_the_token_it_must_predict():
    # THE property. Perturb the LAST position of the input; if an earlier
    # position's output moves, that position read the future.
    def leaks(causal):
        ns: dict = {}
        exec(generate_module(_lm_graph(causal=causal)), ns)  # noqa: S102
        torch.manual_seed(0)
        model = ns["GeneratedModel"]().eval()
        x = torch.randint(0, 20, (1, 16))
        with torch.no_grad():
            before = model(x)
            x2 = x.clone()
            x2[0, -1] = (x[0, -1] + 1) % 20
            after = model(x2)
        return not torch.allclose(before[0, 0], after[0, 0], atol=1e-5)

    assert leaks(False)  # unmasked attention reads the whole window...
    assert not leaks(True)  # ...causal attention does not


def test_inference_and_codegen_agree_on_the_causal_call():
    # Inference probes with the same mask, so a causal node's shape is derived
    # from the identical call the generated code makes.
    shapes, errors = infer_shapes(_lm_graph(causal=True))
    assert not errors
    assert shapes[("blk", "output")] == [1, 16, 32]
    assert shapes[("head", "output")] == [1, 16, 20]  # logits at EVERY position


# --- next-token windows -------------------------------------------------------

def test_windows_pair_a_span_with_itself_shifted_one():
    code = generate_dataloader(Graph(), {"source": "sequence", "block_size": 8, "batch_size": 4})
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    train, val = ns["make_dataloaders"](torch.arange(100))
    assert val is None
    x, y = next(iter(train))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(x[0, 1:], y[0, :-1])  # y is x shifted by one
    assert len(train.dataset) == 100 - 8  # every position is an example


def test_text_is_split_contiguously_so_windows_cannot_leak():
    # Neighbouring windows share all but one token: a random split would put
    # nearly the same text on both sides and the held-out loss would flatter.
    code = generate_dataloader(
        Graph(), {"source": "sequence", "block_size": 4, "val_split": 0.2, "test_split": 0.2})
    assert "random_split" not in code
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    tokens = torch.arange(100)
    train, val, test = ns["make_dataloaders"](tokens, batch_size=8)
    seen = [set(loader.dataset.tokens.tolist()) for loader in (train, val, test)]
    assert seen[0] & seen[1] == set()  # train ∩ val
    assert seen[1] & seen[2] == set()  # val ∩ test
    assert seen[0] & seen[2] == set()  # train ∩ test
    # And they follow the text in order, rather than being sampled from it.
    assert max(seen[0]) < min(seen[1]) < max(seen[1]) < min(seen[2])


def test_a_slice_too_short_for_one_window_yields_no_loader():
    # An empty loader would divide by zero in the loop (the val_split lesson).
    code = generate_dataloader(
        Graph(), {"source": "sequence", "block_size": 32, "val_split": 0.05})
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    _, val = ns["make_dataloaders"](torch.arange(200))  # 5% of 200 = 10 < 32
    assert val is None


# --- the recipe ----------------------------------------------------------------

def test_the_lm_recipe_trains_and_records_perplexity():
    import math

    torch.manual_seed(0)
    text = "the quick brown fox jumps over the lazy dog. " * 60
    vocab = sorted(set(text))
    tokens = torch.tensor([vocab.index(c) for c in text])
    project = _lm_project(vocab=len(vocab), block=16, epochs=3,
                          tokens_cfg={"val_split": 0.1})
    mgr = RunManager()
    assert mgr.start(project, namespace={"tokens": tokens}, emit=lambda m: None) is None
    assert mgr.join(JOIN)
    assert mgr.state == "done", mgr.error

    h = mgr.history
    assert len(h["train_loss"]) == 3 and len(h["val_loss"]) == 3
    # Perplexity is exp(loss) — the number language models are read in.
    assert h["train_perplexity"][0] == pytest.approx(math.exp(h["train_loss"][0]), rel=1e-6)
    assert h["train_perplexity"][-1] < h["train_perplexity"][0]  # it learned


def test_the_lm_loop_scores_every_position():
    # (B, T, V) logits flattened against (B, T) targets — the whole window is
    # supervised, not just its last token.
    src = RECIPES["causal_lm"].generate(_lm_project())
    assert "loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))" in src
    assert "n = y.numel()" in src  # averaged over TOKENS, not batches


def test_unmasked_attention_beats_chance_on_random_tokens():
    # Why the causal check is an error and not advice: on random tokens there
    # is nothing to learn, so a model that improves is reading the answer.
    torch.manual_seed(0)
    tokens = torch.randint(0, 20, (3000,))

    def final_perplexity(causal):
        mgr = RunManager()
        assert mgr.start(_lm_project(causal=causal, epochs=3),
                         namespace={"tokens": tokens}, emit=lambda m: None) is None
        assert mgr.join(JOIN)
        return mgr.history["train_perplexity"][-1]

    causal, leaking = final_perplexity(True), final_perplexity(False)
    assert causal > 15  # ~20 = chance for a 20-token vocabulary
    assert leaking < causal - 2  # the unmasked model "predicts" the unpredictable


# --- diagnostics ----------------------------------------------------------------

def _titles(checks, level=None):
    return " | ".join(c["title"] for c in checks if level is None or c["level"] == level)


def test_unmasked_attention_is_an_error_under_a_next_token_objective():
    tokens = torch.randint(0, 20, (500,))
    errors = _titles(diagnose(_lm_project(causal=False), {"tokens": tokens}), "error")
    assert "TransformerEncoderLayer can see the whole sequence, including the next token" in errors
    # Causal is confirmed rather than merely silent.
    oks = _titles(diagnose(_lm_project(causal=True), {"tokens": tokens}), "ok")
    assert "TransformerEncoderLayer masks the future" in oks


def test_the_head_must_score_the_whole_vocabulary_at_every_position():
    tokens = torch.randint(0, 20, (500,))
    # A head sized to the wrong vocabulary.
    g = _lm_graph(vocab=20, head_out=7)
    project = _lm_project()
    project.models[0].graph = Graph(nodes=g.nodes, edges=g.edges)
    errors = _titles(diagnose(project, {"tokens": tokens}), "error")
    assert "the model outputs 7 logits but the vocabulary is 20" in errors

    # A model that pools the sequence away can't predict per position.
    pooled = graph(
        [
            node("in", "Input", {"shape": "1, 16", "dtype": "long"}),
            node("emb", "Embedding", {"num_embeddings": 20, "embedding_dim": 32}),
            node("blk", "TransformerEncoderLayer", {"nhead": 4, "is_causal": True}),
            node("pool", "Mean", {"dim": 1}),
            node("head", "Linear", {"out_features": 20}),
            node("out", "Output"),
        ],
        [edge("in", "emb"), edge("emb", "blk"), edge("blk", "pool"),
         edge("pool", "head"), edge("head", "out")],
    )
    project.models[0].graph = Graph(nodes=pooled.nodes, edges=pooled.edges)
    errors = _titles(diagnose(project, {"tokens": tokens}), "error")
    assert "not per-position logits" in errors


def test_token_stream_problems_are_reported_before_the_run():
    project = _lm_project()
    # Ids the embedding table can't hold.
    rows = diagnose(project, {"tokens": torch.randint(0, 99, (500,))})
    row = next(c for c in rows if c["level"] == "error" and "holds token ids" in c["title"])
    assert row["title"] == "'tokens' holds token ids 0…98 but the Embedding has 20"
    assert "set num_embeddings to 99" in row["detail"]
    # Floats aren't token ids.
    errors = _titles(diagnose(project, {"tokens": torch.randn(500)}), "error")
    assert "Corpus: 'tokens' is float, but token ids are integers" in errors
    # Nothing registered / nothing picked.
    assert "Corpus: 'tokens' is not registered" in _titles(diagnose(project, {}), "error")
    blank = _lm_project(tokens_cfg={"corpus_var": ""})
    assert "Corpus: nothing picked" in _titles(diagnose(blank, {}), "error")


def test_a_window_longer_than_the_text_is_refused():
    project = _lm_project(block=16, tokens_cfg={"block_size": 400})
    errors = _titles(diagnose(project, {"tokens": torch.randint(0, 20, (300,))}), "error")
    assert "training tokens can't fill a 400-token window" in errors


# --- the tokenizer story: text in, text out ------------------------------------

def test_text_is_a_data_object_the_picker_can_offer():
    from lamplighter.backend.introspect import list_data_variables, variable_kind

    ns = {"corpus": "hello there"}
    assert variable_kind("corpus", ns) == "text"
    entry = list_data_variables(ns)[0]
    assert entry["kind"] == "text"
    assert entry["num_samples"] == 11 and entry["vocab_size"] == 7


def test_a_text_pick_bakes_its_vocabulary_into_the_loader():
    # The vocabulary lives IN the generated source, which is what lets a stored
    # run decode its own samples long after the notebook is gone.
    text = "hello world, hello there. " * 10
    code = generate_dataloader(
        Graph(), {"source": "sequence", "corpus_var": "corpus", "block_size": 8},
        namespace={"corpus": text})
    assert "VOCAB = [" in code and "def decode(ids):" in code
    assert "def make_dataloaders(text, *" in code  # it takes the text itself
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    assert ns["VOCAB"] == sorted(set(text))
    assert ns["decode"](ns["encode"]("hello")) == "hello"  # round-trips
    train, _ = ns["make_dataloaders"](text)
    x, _ = next(iter(train))
    assert ns["decode"](x[0]) in text  # the windows are real slices of the text


def test_ids_are_stable_across_registrations():
    # Sorted vocabulary: re-registering the same text gives the same encoding,
    # so a stored run's ids still mean what they meant.
    from lamplighter.backend.codegen import char_vocab

    assert char_vocab("cba") == char_vocab("abc") == ["a", "b", "c"]


def test_a_pre_tokenized_pick_gets_no_tokenizer():
    code = generate_dataloader(
        Graph(), {"source": "sequence", "corpus_var": "ids", "block_size": 8},
        namespace={"ids": torch.arange(100)})
    assert "VOCAB" not in code and "def make_dataloaders(tokens, *" in code


def test_generation_reads_back_as_text():
    # The whole point of the tokenizer story: a loss curve says how surprised
    # the model is; only a sample says what it writes.
    text = "to be, or not to be, that is the question. " * 60
    vocab = sorted(set(text))
    project = _lm_project(vocab=len(vocab), block=24, epochs=12, lr=3e-3,
                          tokens_cfg={"corpus_var": "corpus", "block_size": 24})
    mgr = RunManager()
    assert mgr.start(project, namespace={"corpus": text}, emit=lambda m: None) is None
    assert mgr.join(JOIN)
    assert mgr.state == "done", mgr.error

    out = mgr.generate(prompt="to be", max_new_tokens=60, temperature=0.5)
    assert out["prompt"] == "to be"
    assert out["text"].startswith("to be")
    assert out["vocab_size"] == len(vocab)
    assert len(out["completion"]) == 60  # one character per requested token
    # Trained to convergence on one repeated sentence, it should reproduce it.
    assert "not to be" in out["text"]


def test_generation_says_why_it_cannot():
    # A run trained on pre-tokenized ids has no vocabulary to read back with.
    tokens = torch.randint(0, 20, (2000,))
    mgr = RunManager()
    assert mgr.start(_lm_project(epochs=1), namespace={"tokens": tokens},
                     emit=lambda m: None) is None
    assert mgr.join(JOIN)
    with pytest.raises(ValueError, match="no vocabulary to read samples back with"):
        mgr.generate(prompt="x")
    # A non-LM run has nothing to generate at all...
    from tests.test_runner import _mlp_graph, _ns

    other = RunManager()
    assert other.start(_mlp_graph({"epochs": 1}), namespace=_ns(), emit=lambda m: None) is None
    assert other.join(JOIN)
    with pytest.raises(ValueError, match="only a language-model run"):
        other.generate()
    # ...and neither does an empty kernel.
    with pytest.raises(ValueError, match="run training first"):
        RunManager().generate()


def test_the_sampler_slides_its_window_and_honors_temperature():
    from lamplighter.backend.codegen import generate_sampling

    src = generate_sampling(block_size=16)
    assert "def generate(model, prompt_ids, *, max_new_tokens=200, temperature=1.0, block_size=16" in src
    assert "ids[-block_size:]" in src  # never more context than it was trained on
    assert "torch.softmax(logits / max(temperature, 1e-6), dim=-1)" in src
    assert "torch.multinomial(probs, num_samples=1)" in src
    assert "@torch.no_grad()" in src


def test_diagnose_names_the_vocabulary_a_text_pick_implies():
    text = "abcdefghij" * 200
    project = _lm_project(vocab=10, block=16, tokens_cfg={"corpus_var": "corpus"})
    rows = diagnose(project, {"corpus": text})
    assert "'corpus': 2000 characters, vocabulary of 10" in _titles(rows, "ok")
    assert "the model covers all 10 characters of 'corpus'" in _titles(rows, "ok")

    # A model whose Embedding can't hold the text says exactly what to set.
    small = _lm_project(vocab=5, block=16, tokens_cfg={"corpus_var": "corpus"})
    row = next(c for c in diagnose(small, {"corpus": text}) if c["level"] == "error")
    assert row["title"] == "'corpus' has 10 distinct characters but the Embedding holds 5"
    assert "set num_embeddings to 10" in row["detail"]


# --- the recipe's data contract -------------------------------------------------

def test_a_language_model_refuses_a_source_it_cannot_read():
    # The failure this prevents is SILENT: fed tabular X/y, the next-token loop
    # trains to completion and reports a perplexity that describes nothing.
    from lamplighter.backend.schema import ModelDef

    g = _lm_graph()
    project = Project(
        models=[ModelDef(id="model", name="LM", graph=Graph(nodes=g.nodes, edges=g.edges))],
        data_nodes=[DataNode(id="d", kind="dataset", config={
            "source": "memory", "x_var": "X", "y_var": "y", "batch_size": 8})],
        links=[ModelLink(id="L", source_data="d", target_model="model")],
        training={"recipe": "causal_lm", "epochs": 1, "device": "cpu"},
    )
    ns = {"X": torch.randn(64, 16), "y": torch.randint(0, 20, (64,))}

    errors = _titles(diagnose(project, ns), "error")
    assert "Language Model (next token) can't train on the 'memory' source" in errors
    # And the runner refuses too — a notebook run never sees diagnose.
    err = RunManager().start(project, namespace=ns, emit=lambda m: None)
    assert err is not None and "can't train on the 'memory' source" in err
    assert "it needs 'sequence'" in err


def test_the_other_recipes_refuse_token_windows():
    # The contract runs both ways: a supervised loop fed (B, T) windows would
    # train something equally meaningless.
    from tests.helpers import single_model_project

    g = _lm_graph()
    project = single_model_project(Graph(nodes=g.nodes, edges=g.edges),
                                   training={"recipe": "supervised"},
                                   data={"source": "sequence", "corpus_var": "corpus"})
    errors = _titles(diagnose(project, {"corpus": "hello world " * 50}), "error")
    assert "Supervised can't train on the 'sequence' source" in errors


def test_an_env_recipe_is_unaffected_by_the_source_contract():
    # RL has no dataset node at all; its own wiring check owns that case.
    from lamplighter.backend.diagnose import source_mismatch
    from lamplighter.backend.recipes import RECIPES

    assert source_mismatch(RECIPES["reinforce"], "memory") is None
    assert source_mismatch(RECIPES["causal_lm"], "sequence") is None


def test_the_window_must_match_the_input_the_model_declares():
    # A PositionalEmbedding sized to the shorter one would index past its table.
    project = _lm_project(block=16, tokens_cfg={"block_size": 64})
    row = next(c for c in diagnose(project, {"tokens": torch.randint(0, 20, (500,))})
               if c["level"] == "error")
    assert row["title"] == "the Input expects 16 tokens but the loader yields windows of 64"
    assert "set the Input shape to (1, 64)" in row["detail"]
    # Agreement is silent.
    ok = _lm_project(block=32, tokens_cfg={"block_size": 32})
    assert not any("the Input expects" in c["title"]
                   for c in diagnose(ok, {"tokens": torch.randint(0, 20, (500,))}))

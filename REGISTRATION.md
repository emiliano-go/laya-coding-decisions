# Registering `coding-decisions` in Laya

Do this **after** the checkpoint passes the gates and is published. The base
checkpoint (`convaiinnovations/laya`) stays the default until then.

## 1. Publish the weights

```sh
python scripts/publish.py \
  --repo emiliano-go/laya-coding-decisions \
  --checkpoint "$LAYACD_DATA/out-coding-decisions" --yes
```

## 2. Register the checkpoint name

`laya/router.py`:

- `DEFAULT_MODELS` — add `"coding-decisions": (BUNDLE_REPO, "coding-decisions")`
  (or point at the standalone repo in `STANDALONE_MODELS`:
  `"coding-decisions": "emiliano-go/laya-coding-decisions"`).
- `_ALIASES` — add `"coding": "coding-decisions"`, `"judge": "coding-decisions"`.

`laya/serve.py`:

- `_KNOWN_MODELS` — add `"coding-decisions"`.
- `_PUBLISHED_MODEL_IDS` — add `"emiliano-go/laya-coding-decisions": "coding-decisions"`.

`nix/laya-serve.nix` — add `coding-decisions` to the model enum.

Then `model: "coding-decisions"` works over `POST /v1/systemone`. Preloading the
name only succeeds once the HF repo exists, so land step 1 first.

## 3. Serving a local checkpoint (optional)

`laya serve` currently accepts only known checkpoint names / published HF ids
(`_resolve_model`), so a freshly trained local directory cannot be served without
publishing. Two options:

- **Encoder shim (works today):** `encoder/script/laya-server.py --model <path>`
  accepts a local path directly. Recommended while iterating.
- **`LAYA_MODEL_PATH` support:** add an env var that registers a local
  checkpoint under a name. Note `Router.__init__` and `load()` call
  `normalise_name()`, which raises for names outside `DEFAULT_MODELS`, so this
  needs either a lenient name path in `Router` or a reserved `"local"` entry in
  the model maps. Add a `tests/test_serve.py` case for it.

## 4. Docs

- `docs/.nav.yml` Fine-tuning list + `docs/index.md` table: link this recipe.
- Add a page `docs/finetune_coding_decisions.md` mirroring `README.md`.
- `tests/test_packaging.py` checks internal links, so keep paths valid.

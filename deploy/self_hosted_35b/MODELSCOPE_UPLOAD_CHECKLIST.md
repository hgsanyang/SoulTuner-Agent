# ModelScope adapter release checklist

Use this checklist before changing
`hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA` from private to public. The release
contains only the PEFT adapter. It must not contain the 72 GB base model,
private datasets, evaluation rows, credentials, or training state.

## Required release files

- [ ] `adapter_model.safetensors`
- [ ] `adapter_config.json`
- [ ] `README.md`
- [ ] `LICENSE`
- [ ] `NOTICE`
- [ ] `SHA256SUMS`
- [ ] No unlisted files, symlinks, subdirectories, or hidden credentials

The expected adapter identity is:

```text
base_model: Qwen/Qwen3.6-35B-A3B
size: 90018600 bytes
sha256: 9a3d2cb5bc2eee3dfc9f7c76c5350509d075aad11b61ddee3b9af2ad90ac272e
```

## Fail-closed local audit

Run the audit against a clean release directory, not the private checkpoint:

```bash
python deploy/self_hosted_35b/audit_public_adapter_repo.py /path/to/release \
  --expected-adapter-sha256 9a3d2cb5bc2eee3dfc9f7c76c5350509d075aad11b61ddee3b9af2ad90ac272e \
  --expected-adapter-size 90018600
```

- [ ] Audit reports `OK`
- [ ] `adapter_config.json` names the exact base model and has
      `inference_mode: true`
- [ ] `SHA256SUMS` lists only `adapter_config.json` and
      `adapter_model.safetensors`
- [ ] Model card contains only aggregate evaluation metrics
- [ ] No raw train, regression, or sealed rows or predictions are present
- [ ] No optimizer, scheduler, RNG, trainer, resume, or environment state is
      present
- [ ] No private paths, tokens, keys, user memory, catalog data, or audio are
      present

## License and provenance gate

- [ ] The official base-model license and pinned base revision are recorded
- [ ] The adapter license and notices are present
- [ ] Every training-data source has a documented right to be used for this
      adapter release
- [ ] Generated/teacher data terms have been reviewed and the decision is
      recorded outside the public repository

The base model is distributed separately. Passing the base-model license check
does not replace the training-data provenance check.

## Clean-download verification

After uploading, use a new empty directory and an authenticated client that
does not print its token:

1. Download the exact repository revision.
2. Record the immutable ModelScope revision.
3. Run the same fail-closed audit on the download.
4. Verify the adapter size and SHA-256 above.
5. Confirm that the model-card loading command resolves the official base and
   the downloaded adapter separately.
6. Inspect the repository file list one last time.

- [ ] Clean download passes
- [ ] Immutable repository revision recorded
- [ ] Repository file list matches the allowlist
- [ ] No credential appeared in logs, screenshots, shell history, or docs

## Visibility change

Changing repository visibility is a separate, user-confirmed action.

- [ ] User reviewed the final file list, license/provenance decision, digest,
      and clean-download result
- [ ] User explicitly approved changing the repository to public
- [ ] Public anonymous download was tested after the change
- [ ] Creation Space was pinned to the audited immutable revision

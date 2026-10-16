"""Failure evidence only; no skipping, retries or tolerance changes."""
import json
import torch


def save_failure(folder, dataset, batch, images, en, de, optimizer, step, last_gradient_norm):
    def describe(feature):
        value = feature.detach().float()
        norms = value.norm(dim=1)
        finite = norms[torch.isfinite(norms)]
        return dict(shape=list(value.shape), dtype=str(feature.dtype),
                    finite=bool(torch.isfinite(value).all()),
                    norm_min=float(finite.min()) if finite.numel() else None,
                    norm_max=float(finite.max()) if finite.numel() else None)
    samples = [dataset.rows[int(index)] for index in batch[2]]
    record = dict(step=step, samples=samples, encoder=[describe(f) for f in en],
                  decoder=[describe(f) for f in de], last_completed_gradient_norm=last_gradient_norm,
                  learning_rates=[g['lr'] for g in optimizer.param_groups])
    (folder / 'numerical_failure.json').write_text(json.dumps(record, indent=2))
    torch.save(dict(images=images, encoder=[f.detach().cpu() for f in en],
                    decoder=[f.detach().cpu() for f in de], metadata=record), folder / 'numerical_failure.pt')

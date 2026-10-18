"""Normal-test object attribution and paired top-20 visual inspection."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

DACL_OBJECTS = ('Bearing', 'Drainage', 'EJoint', 'JTape', 'PEquipment')


def object_masks(dataset, row, profile, source_root):
    if profile == 'dacl_v3_seed42':
        path = dataset.root / row['annotation_path']
        annotation = json.loads(path.read_text(encoding='utf-8'))
        masks = {}
        for label in DACL_OBJECTS:
            mask = Image.new('L', (512, 512), 0)
            draw = ImageDraw.Draw(mask)
            for shape in annotation['shapes']:
                if shape['label'] == label:
                    draw.polygon([tuple(p) for p in shape['points']], fill=255)
            masks[label] = np.asarray(mask) > 0
        return masks, dict(status='covered', annotation=str(path))
    if profile != 's2ds512':
        return None, dict(status='unavailable_no_object_taxonomy')
    if not row['source_sample'].startswith('s2ds_'):
        return None, dict(status='uncovered_external_background')
    ident = row['source_sample'].split('_')[-1]
    matches = list(Path(source_root).glob('*/images/' + ident + '.png'))
    if len(matches) != 1:
        return None, dict(status='uncovered_source_missing_or_ambiguous')
    image_path = matches[0]
    annotation = image_path.parent.parent / 'masks' / (ident + '_lab.png')
    x, y = int(row['crop_x']), int(row['crop_y'])
    box = (x, y, x + 512, y + 512)
    with Image.open(image_path) as im:
        source = np.asarray(im.convert('RGB').crop(box), dtype=np.float32)
    with Image.open(dataset.root / dataset.split / row['image_path']) as im:
        tile = np.asarray(im.convert('RGB'), dtype=np.float32)
    mae = float(np.abs(source - tile).mean())
    if source.shape != tile.shape or mae > 3.0:
        raise ValueError('S2DS source crop alignment failed: ' + row['sample'])
    with Image.open(annotation) as im:
        rgb = np.asarray(im.convert('RGB').crop(box))
    masks = dict(Vegetation=np.all(rgb == (0, 255, 0), axis=2),
                 ControlPoint=np.all(rgb == (0, 0, 255), axis=2))
    return masks, dict(status='covered', annotation=str(annotation), crop=list(box),
                      source_image_mae=mae)


def diagnostics(dataset, profile, paths, labels, before_maps, after_maps,
                before_scores, after_scores, output, source_root):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = dataset.rows
    count = max(1, int(before_maps.shape[-1] ** 2 * .01))
    records, fractions = [], {'before': [], 'after': []}
    for i, row in enumerate(rows):
        if labels[i] != 0:
            continue
        masks, provenance = object_masks(dataset, row, profile, source_root)
        record = dict(image_path=paths[i], **provenance)
        if masks is not None:
            masks['union'] = np.logical_or.reduce(list(masks.values()))
            record['object_area_fraction_native512'] = float(masks['union'].mean())
            for method, maps in [('before', before_maps), ('after', after_maps)]:
                # Fixed-size deterministic top set; ties resolved by pixel index.
                top = np.argsort(-maps[i].ravel(), kind='stable')[:count]
                values = {}
                for name, mask in masks.items():
                    resized = np.asarray(Image.fromarray(mask).resize((256, 256), Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST))
                    values[name] = float(resized.ravel()[top].mean())
                record[method] = values
                area = float(np.asarray(Image.fromarray(masks['union']).resize((256, 256), Image.NEAREST)).mean())
                record['random_baseline_area_fraction_256'] = area
                record.setdefault('enrichment', {})[method] = values['union'] / area if area > 0 else None
                fractions[method].append(values['union'])
        records.append(record)
    normal = np.flatnonzero(labels == 0)
    rankings = {}
    for method, scores in [('before', before_scores), ('after', after_scores)]:
        rankings[method] = normal[np.argsort(-scores[normal], kind='stable')[:20]].tolist()
    chosen = sorted(set(rankings['before']) | set(rankings['after']))
    # One shared color range for before/after; avoid per-panel normalization.
    low = float(min(before_maps[normal].min(), after_maps[normal].min()))
    high = float(max(before_maps[normal].max(), after_maps[normal].max()))
    import matplotlib.cm as cm
    for i in chosen:
        with Image.open(paths[i]) as im:
            original = im.convert('RGB').resize((512, 512))
        panels = [original]
        for maps in (before_maps, after_maps):
            normalized = np.clip((maps[i] - low) / max(high - low, 1e-8), 0, 1)
            color = Image.fromarray((cm.get_cmap('jet')(normalized)[..., :3] * 255).astype('uint8')).resize((512, 512))
            panels.append(Image.blend(original, color, .45))
        canvas = Image.new('RGB', (1536, 540), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, panel in enumerate(panels):
            canvas.paste(panel, (j * 512, 28))
        draw.text((8, 5), 'Original | V0 | CRMC   sample=' + rows[i]['sample'], fill='black')
        canvas.save(output / ('normal_%04d.png' % i))
    containing = [r for r in records if r.get('random_baseline_area_fraction_256', 0) > 0]
    object_only = dict(images=len(containing), mean_overlap={}, mean_enrichment={})
    for method in ('before', 'after'):
        object_only['mean_overlap'][method] = float(np.mean([r[method]['union'] for r in containing])) if containing else None
        object_only['mean_enrichment'][method] = float(np.mean([r['enrichment'][method] for r in containing])) if containing else None
    report = dict(object_containing_normals=object_only, top_fraction=.01, top_pixels=count, normals=len(normal),
                  covered_normals=len(fractions['before']),
                  uncovered_normals=len(normal) - len(fractions['before']),
                  records=records, top20_indices=rankings,
                  overlay_scale=dict(low=low, high=high, shared=True),
                  mean_union_fraction={k: float(np.mean(v)) if v else None for k, v in fractions.items()},
                  note='Descriptive diagnostic only; conditional on annotation coverage; lower overlap alone does not prove lower absolute false positives')
    (output / 'object_attribution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

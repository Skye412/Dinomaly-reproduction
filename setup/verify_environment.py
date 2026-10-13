import json
import platform
from pathlib import Path
import torch
import torchvision
import dinomaly_mvtec_uni
import dinomaly_visa_uni
from dataset import MVTecDataset, get_data_transforms
from torchvision.datasets import ImageFolder

root = Path('/home/skye/data/Skye')
transform, gt = get_data_transforms(448, 392)
report = {'python': platform.python_version(), 'torch': torch.__version__,
          'torchvision': torchvision.__version__, 'cuda_runtime': torch.version.cuda,
          'cuda_available': torch.cuda.is_available(), 'training_entry_imports': 'passed'}
assert torch.cuda.is_available(), 'CUDA unavailable'
report['gpu'] = torch.cuda.get_device_name(0)
x = torch.randn(256, 256, device='cuda:0')
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
report['cuda_matmul'] = 'passed'
for name, path in [('mvtec', root / 'databases/MVTEC'), ('visa', root / 'databases/VISA_pytorch/1cls')]:
    counts = {'train': 0, 'test_good': 0, 'test_bad': 0}
    for category in sorted(p for p in path.iterdir() if p.is_dir()):
        train = ImageFolder(str(category / 'train'), transform=transform)
        test = MVTecDataset(str(category), transform, gt, 'test')
        labels = list(test.labels)
        counts['train'] += len(train)
        counts['test_good'] += labels.count(0)
        counts['test_bad'] += labels.count(1)
        assert len(test.img_paths) == len(test.gt_paths)
        a = train[0][0]
        assert tuple(a.shape) == (3, 392, 392)
        for index in [labels.index(0), labels.index(1)]:
            test[index]
    report[name] = counts
assert report['mvtec'] == dict(train=3629, test_good=467, test_bad=1258)
assert report['visa'] == dict(train=8659, test_good=962, test_bad=1200)
Path('reproduction/environment_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

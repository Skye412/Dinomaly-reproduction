from pathlib import Path
import subprocess

repo = Path('/home/skye/data/Skye/Dinomaly')
out = repo / 'reproduction'
out.mkdir(exist_ok=True)
for filename in ['dinomaly_mvtec_uni.py', 'dinomaly_visa_uni.py']:
    p = repo / filename
    content = p.read_text()
    content = content.replace("device = 'cuda:1'", "device = 'cuda:0'")
    p.write_text(content)
(out / 'server.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=str(repo), text=True))

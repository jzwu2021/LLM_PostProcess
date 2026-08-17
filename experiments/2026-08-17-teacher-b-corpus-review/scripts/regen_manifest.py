import subprocess, os
d = '/media/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review'
os.chdir(d)
files = []
for root, dirs, fs in os.walk('.'):
    dirs[:] = [x for x in dirs if x != '__pycache__']
    for f in fs:
        p = os.path.join(root, f)[2:]
        if f == 'MANIFEST.sha256':
            continue
        files.append(p)
files.sort()
with open('MANIFEST.sha256', 'w') as out:
    for i in range(0, len(files), 200):
        r = subprocess.run(['sha256sum'] + files[i:i+200], capture_output=True, text=True)
        out.write(r.stdout)
print('FILES', len(files))
r = subprocess.run(['sha256sum', '-c', 'MANIFEST.sha256'], capture_output=True, text=True)
bad = [l for l in r.stdout.splitlines() if not l.endswith(': OK')]
print('MANIFEST_CHECK', 'PASS' if (r.returncode == 0 and not bad) else 'FAIL', bad[:5])

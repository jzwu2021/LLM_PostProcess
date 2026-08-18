import re
p = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/scripts/tb_gen_batch_0203.py"
s = open(p).read()
new = open(p.replace("tb_gen_batch_0203.py", "tb_stances_0203.txt")).read()
i = s.index("STANCES = [")
j = s.index("]\n\nCRITIQUE")
s = s[:i] + "STANCES = [\n" + new + s[j:]
open(p, "w").write(s)
print("spliced")

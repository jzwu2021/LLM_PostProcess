import re
src = "/home/johnson/workspace/LLM_PostProcess/scripts/tb_verify_batch_0139.py"
dst = "/home/johnson/workspace/LLM_PostProcess/scripts/tb_verify_batch_0140.py"
c = open(src, encoding="utf-8").read().replace("0139", "0140")
open(dst, "w", encoding="utf-8").write(c)
print("ok")

"""分析对比文档中某题的检索切片内容"""
import sys, re

with open("eval/output/answer_comparison_20260614_112948.md", encoding="utf-8") as f:
    content = f.read()

qid = sys.argv[1] if len(sys.argv) > 1 else "1"
q_start = content.find(f"## Q{qid} ")
next_start = content.find(f"## Q{int(qid)+1} ")
q_section = content[q_start:next_start]

# 找所有 <details> 块
parts = q_section.split("<details>")
print(f"Q{qid}: {len(parts)-1} chunks")
print()

for pi, p in enumerate(parts):
    if pi == 0:
        continue
    # 摘要行
    sm_start = p.find("<summary>") + 9
    sm_end = p.find("</summary>")
    summary = p[sm_start:sm_end].strip()

    # 提取代码块文本
    code_marker = "`" * 3
    c1 = p.find(code_marker)
    c2 = p.find(code_marker, c1 + 3)
    text = p[c1+3:c2].strip()

    # 高亮关键词
    kw_highlight = ["9.0.2", u"荷载设计值", u"荷载标准值", u"承载力",
                    "CG", u"γ0", u"永久荷载", u"可变荷载", u"极限状态"]
    found_kw = [k for k in kw_highlight if k in text]

    print(f"--- #{pi} [{summary[:60]}] ---")
    if found_kw:
        print(f"  **HIT KEYWORDS**: {found_kw}")
    print(text[:400])
    print()

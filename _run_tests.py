"""跨域检索测试 — 4域 × 3题"""
import json, urllib.request, time

tests = [
    ("B01", "3~110kV高压配电装置的设计应符合哪些基本要求", "变电", "GB 50060"),
    ("B02", "变电站接地装置的施工和验收标准是什么", "变电", "GB 50169"),
    ("B03", "变电所的消防设计要求有哪些", "变电", "消防"),
    ("P04", "10kV配电线路的安全距离是多少", "配电", "配电线路"),
    ("P05", "配电变压器的选用有什么技术要求", "配电", "变压器"),
    ("P06", "配电网的电压偏差允许范围是多少", "配电", "电压偏差"),
    ("S07", "66kV及以下架空电力线路杆塔结构的承载力设计应采用什么极限状态", "送电输电", "GB 50061"),
    ("S08", "架空输电线路的电磁环境限值有哪些规定", "送电输电", "电磁环境"),
    ("S09", "电力电缆线路的敷设有什么要求", "送电输电", "GB 50217"),
    ("Z10", "福建省电力公司对变电消防管理有什么发文要求", "综合", "闽电"),
    ("Z11", "国家电网公司对配电自动化有什么规定", "综合", "国家电网"),
    ("Z12", "榕能公司对设计管理有什么工作要求", "综合", "榕能"),
]

results = []
for tid, query, expected_domain, expected_key in tests:
    print(f"[{tid}] {query[:45]}...")
    try:
        req = urllib.request.Request(
            "http://localhost:8000/search",
            data=json.dumps({"query": query, "top_k": 10}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())

        top3_files = []
        top3_domains = []
        top3_categories = []
        for r in data.get("results", [])[:3]:
            fp = r.get("file_path", "")
            # extract just filename
            for sep in ("/", "\\"):
                if sep in fp:
                    fp = fp.split(sep)[-1]
            top3_files.append(fp[:60])
            top3_domains.append(r.get("domain", "?"))
            top3_categories.append(r.get("category", "?"))

        hit_expected = any(
            expected_key.lower() in str(s).lower() for s in top3_files
        )

        r = {
            "id": tid,
            "query": query,
            "expected_domain": expected_domain,
            "expected_key": expected_key,
            "retrieved_domain": data.get("domain"),
            "query_type": data.get("query_type"),
            "candidates": data.get("total_candidates", 0),
            "elapsed_ms": data.get("elapsed_ms", 0),
            "top1_file": top3_files[0] if top3_files else "",
            "top1_domain": top3_domains[0] if top3_domains else "?",
            "top1_category": top3_categories[0] if top3_categories else "?",
            "top3_files": " | ".join(top3_files),
            "top3_domains": " | ".join(top3_domains),
            "hit_expected": hit_expected,
        }
        results.append(r)
        print(
            f"  domain={r['retrieved_domain']}, type={r['query_type']}, "
            f"candidates={r['candidates']}, top1={r['top1_file'][:45]}"
        )
    except Exception as e:
        results.append({"id": tid, "query": query, "error": str(e)})
        print(f"  ERROR: {e}")
    time.sleep(2)

# Summary
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
domain_hits = {d: 0 for d in ["变电", "配电", "送电输电", "综合"]}
domain_total = {d: 0 for d in ["变电", "配电", "送电输电", "综合"]}
category_correct = 0
key_hits = 0

for r in results:
    if "error" in r:
        print(f"[{r['id']}] ERROR: {r['error']}")
        continue
    dom = r["expected_domain"]
    domain_total[dom] += 1
    dom_match = r["retrieved_domain"] == r["expected_domain"]
    if dom_match:
        domain_hits[dom] += 1
    if r["hit_expected"]:
        key_hits += 1
    if r["top1_category"] == "标准规范":
        category_correct += 1  # all our test questions expect 标准规范 category

    dom_icon = "V" if dom_match else ("~" if r["retrieved_domain"] is None else "X")
    key_icon = "V" if r["hit_expected"] else "X"
    print(
        f"[{r['id']}] dom={dom_icon} key={key_icon} "
        f"query_type={r['query_type']} "
        f"top1_domain={r['top1_domain']:6s} top1_cat={r['top1_category']:6s} "
        f"file={r['top1_file'][:50]}"
    )

print()
print("--- Domain Accuracy ---")
for d in ["变电", "配电", "送电输电", "综合"]:
    print(f"  {d}: {domain_hits[d]}/{domain_total[d]}")
print(f"  Overall: {sum(domain_hits.values())}/{sum(domain_total.values())}")
print(f"--- Key file hits: {key_hits}/{len([r for r in results if 'error' not in r])} ---")
print(f"--- Category=标准规范: {category_correct}/{len([r for r in results if 'error' not in r])} ---")

with open("E:/RongNengRAG/_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nResults saved to _test_results.json")

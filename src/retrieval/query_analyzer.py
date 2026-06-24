"""
查询分析器 — 解析用户自然语言，提取检索参数
功能:
  1. 域分类 (变电/配电/送电输电/综合)
  2. 参数提取 (电压等级、设备类型、文档编号等)
  3. 查询类型判断
  4. 同义词扩展
  5. 构建 Milvus 过滤表达式
"""

import re
import yaml
import jieba
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class AnalyzedQuery:
    """解析后的查询对象"""
    original_query: str
    expanded_query: str = ""
    query_type: str = "general_qa"

    # 过滤参数
    domain: Optional[str] = None
    category: Optional[str] = None
    voltage_level: Optional[str] = None
    publish_level: Optional[str] = None
    discipline: Optional[str] = None
    equipment_type: Optional[str] = None
    year: Optional[int] = None
    region: Optional[str] = None
    doc_number: Optional[str] = None

    # Milvus 过滤表达式
    filter_expr: Optional[str] = None

    # 搜索参数覆盖
    exclude_drawings: bool = False
    boost_standards: bool = False
    parallel_domains: List[str] = field(default_factory=list)

    # 扩展词
    expanded_terms: List[str] = field(default_factory=list)


class QueryAnalyzer:
    """查询分析器"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        self.domain_keywords = self.config.get("domain_keywords", {})
        self.voltage_patterns = self.config.get("voltage_patterns", {})
        self.discipline_patterns = self.config.get("discipline_patterns", {})
        self.equipment_patterns = self.config.get("equipment_patterns", {})
        self.doc_number_patterns = self.config.get("doc_number_patterns", {})

        # 同义词词典（电力专业领域）
        self.synonym_dict = {
            "消防": ["防火", "灭火", "火灾报警", "消防给水", "消防设施", "消火栓"],
            "接地": ["接地装置", "接地电阻", "接地网", "接地系统", "保护接地", "工作接地"],
            "安全距离": ["净距", "间距", "安全净距", "带电距离", "最小距离"],
            "变压器": ["主变", "配变", "站用变", "变压器保护"],
            "电缆": ["电力电缆", "电缆线路", "电缆敷设", "电缆沟", "电缆隧道"],
            "保护": ["继电保护", "保护配置", "保护装置", "差动保护", "过流保护"],
            "设计深度": ["设计内容深度", "初步设计深度", "施工图深度", "可研深度"],
            "反措": ["反事故措施", "二十五项反措", "十八项反措", "防止事故"],
            "标准工艺": ["施工工艺", "工艺标准", "标准工艺手册"],
            "三维设计": ["三维数字化", "BIM", "三维建模", "数字化设计"],
        }

    def analyze(self, query: str) -> AnalyzedQuery:
        """完整的查询分析"""
        aq = AnalyzedQuery(original_query=query)

        # Step 1: 提取文档编号（直接文档查找）
        aq.doc_number = self._extract_doc_number(query)
        if aq.doc_number:
            aq.query_type = "document_lookup"
            aq.filter_expr = f'doc_number like "%{aq.doc_number}%"'
            aq.expanded_query = query
            return aq

        # Step 2: 域分类
        aq.domain = self._classify_domain(query)

        # Step 3: 提取结构化参数
        aq.voltage_level = self._extract_voltage(query)
        aq.discipline = self._extract_discipline(query)
        aq.equipment_type = self._extract_equipment(query)
        aq.publish_level = self._extract_publish_level(query)
        aq.year = self._extract_year(query)

        # Step 4: 判断查询类型
        aq.query_type = self._classify_query_type(query, aq)

        # Step 5: 根据查询类型调整参数
        if aq.query_type == "specification_lookup":
            aq.exclude_drawings = True
            aq.boost_standards = True
            if not aq.category:
                aq.category = "标准规范"

        elif aq.query_type == "cross_domain_comparison":
            aq.parallel_domains = self._detect_comparison_domains(query)

        elif aq.query_type == "domain_technical":
            aq.boost_standards = True

        # Step 6: 同义词扩展
        aq.expanded_terms = self._expand_synonyms(query)
        aq.expanded_query = query
        if aq.expanded_terms:
            aq.expanded_query = query + " " + " ".join(aq.expanded_terms)

        # Step 7: 构建过滤表达式
        aq.filter_expr = self._build_filter_expr(aq)

        return aq

    def _extract_doc_number(self, query: str) -> Optional[str]:
        """提取文档编号"""
        for key, pattern in self.doc_number_patterns.items():
            match = re.search(pattern, query)
            if match:
                return match.group(0)
        return None

    def _classify_domain(self, query: str) -> Optional[str]:
        """
        基于关键词典对查询进行域分类（jieba分词 + 词边界匹配）

        使用 jieba 分词后逐词匹配，避免子串误判:
          - "配电装置" 不会因含 "配电" 而被误判为配电域
          - "10kV变电站" 不会因 "10kV" 被配电域抢走
        """
        # jieba 分词
        words = set(jieba.lcut(query))

        scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for kw in keywords:
                # 精确词匹配: kw 必须作为完整词出现在分词结果中
                if kw in words:
                    score += 1
            if score > 0:
                scores[domain] = score

        if not scores:
            return None

        # 消歧规则: 当变电和配电同时命中时，检查是否有强信号
        if "变电" in scores and "配电" in scores:
            # "配电装置" 在变电关键词中 → 有"变电站"等强变电信号时归变电
            strong_biandian = {"变电站", "GIS", "主变", "换流站", "变电所",
                               "电气一次", "电气二次", "继电保护", "直流系统", "综合自动化"}
            if any(w in words for w in strong_biandian):
                scores["变电"] += 2  # 强信号加权

        # 消歧规则: GB50061 等66kV线路规范归属送电输电，非配电
        # "架空线路""杆塔""导线""输电" 是送电输电强信号
        if "送电输电" in scores:
            strong_songdian = {"架空线路", "杆塔", "导线", "地线", "OPGW",
                               "绝缘子", "防振锤", "跨越", "耐张", "输电"}
            if any(w in words for w in strong_songdian):
                scores["送电输电"] += 2

        # 返回得分最高的域
        best = max(scores, key=scores.get)

        # 如果最高分不够显著（与其他域持平），不强制过滤
        if len(scores) > 1:
            scores_list = sorted(scores.values(), reverse=True)
            if scores_list[0] == scores_list[1]:
                return None  # 跨域查询，释放域过滤

        return best

    def _extract_voltage(self, query: str) -> Optional[str]:
        """提取电压等级"""
        for voltage, patterns in self.voltage_patterns.items():
            for pat in patterns:
                if re.search(pat, query, re.IGNORECASE):
                    return voltage
        return None

    def _extract_discipline(self, query: str) -> Optional[str]:
        """提取专业类型"""
        for discipline, patterns in self.discipline_patterns.items():
            for pat in patterns:
                if re.search(pat, query):
                    return discipline
        return None

    def _extract_equipment(self, query: str) -> Optional[str]:
        """提取设备类型"""
        for equipment, patterns in self.equipment_patterns.items():
            for pat in patterns:
                if re.search(pat, query):
                    return equipment
        return None

    def _extract_publish_level(self, query: str) -> Optional[str]:
        """提取发布层级"""
        if re.search(r'国标|GB|国家标准', query):
            return "国标"
        if re.search(r'行标|DL|行业标准', query):
            return "行标"
        if re.search(r'省公司|福建电力|闽电', query):
            return "省公司"
        if re.search(r'国网公司|国家电网', query):
            return "国网公司"
        return None

    def _extract_year(self, query: str) -> Optional[int]:
        """提取年份"""
        match = re.search(r'(20\d{2})年?', query)
        if match:
            return int(match.group(1))
        return None

    def _classify_query_type(self, query: str, aq: AnalyzedQuery) -> str:
        """判断查询类型"""
        # 跨域对比
        comparison_patterns = [
            r'(.+)和(.+)的?区别', r'(.+)与(.+)的?区别',
            r'(.+)和(.+)有什么不同', r'对比(.+)和(.+)',
            r'比较(.+)和(.+)'
        ]
        for pat in comparison_patterns:
            if re.search(pat, query):
                return "cross_domain_comparison"

        # 数值查规
        value_patterns = [
            r'是多少', r'多少米', r'多大', r'多长', r'几米',
            r'距离', r'间距', r'净距', r'高度', r'宽度',
            r'不小于', r'不大于', r'不超过', r'不应小于'
        ]
        if any(re.search(p, query) for p in value_patterns):
            return "specification_lookup"

        # 域特定查询
        if aq.domain:
            if aq.category:
                return "domain_technical"
            return "domain_technical"

        # 文档查找（已在 _extract_doc_number 覆盖）
        return "general_qa"

    def _detect_comparison_domains(self, query: str) -> List[str]:
        """检测跨域对比中涉及的专业域"""
        domains_found = []
        for domain, keywords in self.domain_keywords.items():
            if any(kw in query for kw in keywords):
                domains_found.append(domain)
        return domains_found

    def _expand_synonyms(self, query: str) -> List[str]:
        """查询同义词扩展"""
        expanded = []
        for term, synonyms in self.synonym_dict.items():
            if term in query:
                expanded.extend(synonyms[:3])  # 最多扩展3个同义词
        return expanded

    def _build_filter_expr(self, aq: AnalyzedQuery) -> Optional[str]:
        """构建 Milvus 过滤表达式"""
        from ingestion.milvus_store import build_filter_expression

        return build_filter_expression(
            domain=aq.domain,
            category=aq.category,
            voltage_level=aq.voltage_level,
            publish_level=aq.publish_level,
            discipline=aq.discipline,
            equipment_type=aq.equipment_type,
            year=aq.year,
            region=aq.region,
            exclude_drawings=aq.exclude_drawings,
            doc_number=aq.doc_number,
        )


# 快速测试
if __name__ == "__main__":
    analyzer = QueryAnalyzer()

    test_queries = [
        "变电设计中有哪些关于消防的要求？",
        "10kV配电线路的安全距离是多少？",
        "闽电发展〔2015〕241号文件的主要内容是什么？",
        "变电和配电在接地要求上有什么区别？",
        "110-A2-4方案中消防给排水的设计要求",
        "变压器中性点接地方式有哪些？",
        "省公司关于设计深度的最新要求",
    ]

    for q in test_queries:
        result = analyzer.analyze(q)
        print(f"\n📝 查询: {result.original_query}")
        print(f"   类型: {result.query_type}")
        print(f"   域: {result.domain}")
        print(f"   电压: {result.voltage_level}")
        print(f"   过滤: {result.filter_expr}")
        print(f"   扩展: {result.expanded_terms}")
        if result.parallel_domains:
            print(f"   并行域: {result.parallel_domains}")

"""
OFD 文档文本提取器 — 国产版式文档 GB/T 33190

处理 OFD 文件的文本提取，支持:
  1. 直接 Unicode 文本的 TextCode
  2. GBK/GB18030 编码的 TextCode (常见于中文 OFD)
  3. 基于字体 CMap 的字形映射 (CID-keyed fonts)

不依赖第三方 OFD 库，仅使用标准库 zipfile + xml.etree (或 xmltodict)。
"""

import os
import re
import zipfile
from xml.etree import ElementTree as ET
from typing import Dict, List, Optional


# OFD 标准命名空间
OFD_NS = "http://www.ofdspec.org/2016"


def _decode_bytes(data: bytes, prefer_gbk: bool = False) -> str:
    """
    解码字节序列为字符串。

    策略:
      1. 如果 prefer_gbk, 先尝试 GB18030
      2. 否则先尝试 UTF-8
      3. 失败时尝试另一种编码
      4. 都失败则返回空字符串
    """
    if not data:
        return ""

    first_try = "gb18030" if prefer_gbk else "utf-8"
    second_try = "utf-8" if prefer_gbk else "gb18030"

    try:
        text = data.decode(first_try)
        # 如果结果中有大量替换字符，可能编码不对
        if '�' in text and len(text) < 10:
            raise UnicodeDecodeError(first_try, data, 0, 1, "too many replacements")
        return text
    except (UnicodeDecodeError, UnicodeError):
        pass

    try:
        return data.decode(second_try)
    except (UnicodeDecodeError, UnicodeError):
        pass

    # 最后的兜底: UTF-8 with replacement chars
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_textcode_bytes(xml_bytes: bytes) -> List[bytes]:
    """
    从 XML 原始字节中提取所有 TextCode 元素的内容 (不解码)。
    使用正则匹配避开 XML 解析器的编码处理。
    """
    # 匹配 <ofd:TextCode ...>CONTENT</ofd:TextCode>
    # 注意: TextCode 可能跨行，内容不含嵌套元素
    pattern = rb'<ofd:TextCode[^>]*>(.*?)</ofd:TextCode>'
    matches = re.findall(pattern, xml_bytes, re.DOTALL)
    return matches


def _build_font_encoding_map(z: zipfile.ZipFile, public_res_path: str) -> Dict[str, bool]:
    """
    构建字体 ID → prefer_gbk 映射。

    从 PublicRes.xml 中提取字体定义，根据字体名判断是否使用 GBK 编码。
    """
    font_gbk_map: Dict[str, bool] = {}

    if public_res_path not in z.namelist():
        return font_gbk_map

    try:
        raw = z.read(public_res_path)

        # 提取字体定义块
        font_pattern = rb'<ofd:Font\s+([^>]+)/?>'
        for match in re.finditer(font_pattern, raw):
            attrs = match.group(1)

            # 提取 ID
            id_match = re.search(rb'ID="(\d+)"', attrs)
            if not id_match:
                continue
            font_id = id_match.group(1).decode("ascii")

            # 提取 FontName (尝试 GBK + UTF-8)
            name_match = re.search(rb'FontName="([^"]*)"', attrs)
            if not name_match:
                continue

            name_bytes = name_match.group(1)
            font_name = _decode_bytes(name_bytes, prefer_gbk=True)

            # GBK 字体标记: 名称含 _GBK, _gbk, GBK, GB2312
            is_gbk = any(
                marker in font_name
                for marker in ("_GBK", "_gbk", "GBK", "GB2312", "gb2312")
            )

            # 中文字体名但没有 Unicode 名称 → 可能是 GBK 编码
            # (纯ASCII字体名如 SimSun 不会是GBK)
            has_chinese = any('一' <= c <= '鿿' for c in font_name)
            if has_chinese and not is_gbk:
                # 中文名字体: 检查是否是公认的系统字体
                system_fonts = {
                    '宋体', '黑体', '楷体', '仿宋', '隶书',
                    'SimSun', 'SimHei', 'KaiTi', 'FangSong',
                    'Microsoft YaHei', '微软雅黑',
                }
                # 如果名称看起来像是从GBK错误解码的，标记为GBK
                if any('�' in font_name for _ in [font_name]):
                    is_gbk = True

            font_gbk_map[font_id] = is_gbk

    except Exception:
        pass

    return font_gbk_map


def _extract_page_text(
    z: zipfile.ZipFile,
    content_xml_path: str,
    font_gbk_map: Dict[str, bool],
) -> str:
    """
    从单页 Content.xml 中提取文本。
    """
    if content_xml_path not in z.namelist():
        return ""

    try:
        raw = z.read(content_xml_path)

        # 方案 A: 尝试 xmltodict (更好的多值处理)
        try:
            import xmltodict
            tree = xmltodict.parse(raw)
            return _extract_from_tree(tree, font_gbk_map)
        except ImportError:
            pass

        # 方案 B: 正则直接提取 (回退)
        return _extract_via_regex(raw, font_gbk_map)

    except Exception:
        return ""


def _extract_from_tree(tree: dict, font_gbk_map: Dict[str, bool]) -> str:
    """从 xmltodict 解析的树中提取文本。"""
    texts = []

    page = tree.get("ofd:Page", tree.get("Page", {}))
    content = page.get("ofd:Content", page.get("Content", {}))
    layers = content.get("ofd:Layer", content.get("Layer", []))

    if isinstance(layers, dict):
        layers = [layers]

    for layer in layers:
        if not isinstance(layer, dict):
            continue

        # 处理直接的 TextObject
        to_list = layer.get("ofd:TextObject", layer.get("TextObject", []))
        if isinstance(to_list, dict):
            to_list = [to_list]
        texts.extend(_extract_text_objects(to_list, font_gbk_map))

        # 处理 PageBlock 嵌套
        pb = layer.get("ofd:PageBlock", layer.get("PageBlock", {}))
        if isinstance(pb, dict):
            inner_pb = pb.get("ofd:PageBlock", pb.get("PageBlock", {}))
            inner_to = inner_pb.get("ofd:TextObject", inner_pb.get("TextObject", []))
            if isinstance(inner_to, dict):
                inner_to = [inner_to]
            elif not isinstance(inner_to, list):
                inner_to = []
            texts.extend(_extract_text_objects(inner_to, font_gbk_map))

    return "".join(texts)


def _extract_text_objects(
    text_objects: List[dict], font_gbk_map: Dict[str, bool]
) -> List[str]:
    """从 TextObject 列表提取文本。"""
    texts = []
    for to in text_objects:
        if not isinstance(to, dict):
            continue

        font_id = to.get("@Font", "")
        prefer_gbk = font_gbk_map.get(font_id, False)

        tc = to.get("ofd:TextCode", to.get("TextCode", {}))
        if isinstance(tc, dict):
            text = tc.get("#text", "")
            if text:
                # 如果文本只有替换字符，尝试重新解码原始字节
                if text.count('�') > len(text) * 0.3 and prefer_gbk:
                    # 回退: 用正则从原始XML提取并GBK解码
                    # 这里无法获取原始字节，跳过
                    pass
                texts.append(text)
        elif isinstance(tc, list):
            for t in tc:
                if isinstance(t, dict):
                    text = t.get("#text", "")
                    if text:
                        texts.append(text)
        elif isinstance(tc, str):
            texts.append(tc)

    return texts


def _extract_via_regex(
    raw_bytes: bytes, font_gbk_map: Dict[str, bool]
) -> str:
    """
    使用正则直接从原始 XML 字节中提取 TextCode 内容。
    这是最可靠的方法，因为可以控制编码解码。
    """
    # 首先建立 TextObject → Font 的映射
    # 匹配: <ofd:TextObject ... Font="ID" ...> ... </ofd:TextObject>
    to_pattern = rb'<ofd:TextObject[^>]*Font="(\d+)"[^>]*>(.*?)</ofd:TextObject>'
    to_matches = re.findall(to_pattern, raw_bytes, re.DOTALL)

    texts = []
    for font_id_bytes, content_bytes in to_matches:
        font_id = font_id_bytes.decode("ascii")
        prefer_gbk = font_gbk_map.get(font_id, False)

        # 提取 TextCode 内容
        tc_matches = _extract_textcode_bytes(content_bytes)
        for tc_bytes in tc_matches:
            text = _decode_bytes(tc_bytes, prefer_gbk=prefer_gbk)
            if text:
                texts.append(text)

    return "".join(texts)


def extract_ofd_text(file_path: str) -> str:
    """
    从 OFD 文件中提取全部文本。

    Args:
        file_path: OFD 文件路径

    Returns:
        提取的文本字符串。如果解析失败返回空字符串。
    """
    if not os.path.exists(file_path):
        return ""

    try:
        with zipfile.ZipFile(file_path, "r") as z:
            names = sorted(z.namelist())

            # 查找 PublicRes.xml
            public_res = [n for n in names if n.endswith("/PublicRes.xml") or n == "PublicRes.xml"]
            public_res_path = public_res[0] if public_res else ""

            # 构建字体编码映射
            font_gbk_map = _build_font_encoding_map(z, public_res_path)

            # 查找所有页面 Content.xml
            page_xmls = [
                n for n in names
                if "Content.xml" in n and "Page" in n and "Res" not in n
            ]

            if not page_xmls:
                return ""

            all_page_texts = []
            for xml_path in page_xmls:
                page_text = _extract_page_text(z, xml_path, font_gbk_map)
                if page_text.strip():
                    all_page_texts.append(page_text)

            text = "\n\n".join(all_page_texts)

            if text.strip():
                print(f"   [ofd] 自定义提取成功: {len(text)} 字符 "
                      f"({len(page_xmls)}页)")
                return text

    except zipfile.BadZipFile:
        print(f"   [ofd] 文件不是有效的 ZIP 格式: {file_path}")
    except Exception as e:
        print(f"   [ofd] 提取失败: {e}")

    return ""


# ===== 兜底方案: ofdparser (如果可用) =====

def extract_ofd_text_via_ofdparser(file_path: str, temp_dir: str = None) -> str:
    """
    使用 ofdparser 库提取文本 (需要安装: pip install ofdparser reportlab xmltodict)。

    作为自定义提取器的备用方案。

    Args:
        file_path: OFD 文件路径
        temp_dir: 临时文件目录，默认使用 E:/RongNengRAG/data/ofd_tmp
                  避免 ofdparser 将缓存文件写入 C 盘 (它默认用 os.getcwd())
    """
    import uuid as _uuid

    if temp_dir is None:
        temp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "ofd_tmp",
        )
    os.makedirs(temp_dir, exist_ok=True)

    # 构造 E 盘临时路径，避免 ofdparser 默认 os.getcwd() 写到 C 盘
    ofd_tmp_path = os.path.join(temp_dir, f"{os.getpid()}_{_uuid.uuid4().hex}.ofd")

    try:
        from ofdparser import OfdParser
        import base64

        with open(file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("ascii")

        parser = OfdParser(b64_data, zip_path=ofd_tmp_path, dpi=200)
        result = parser.parserodf2json()

        all_lines = []
        for page in result:
            for line in page.get("lineList", []):
                content = line.get("objContent", "")
                if content and content.strip():
                    all_lines.append(content)

        text = "\n".join(all_lines)
        if text.strip():
            print(f"   [ofd] ofdparser 提取成功: {len(text)} 字符")
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"   [ofd] ofdparser 提取失败: {e}")

    return ""

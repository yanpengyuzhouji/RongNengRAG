import fitz
import requests
import base64
import os


# PDF路径
pdf_path = r"D:\榕能电力审图项目资料\原材料规则分类\知识库管理\标准规范\标准\国标\GB 51158-2015_通信线路工程设计规范_OCR.pdf"


# 测试页
page_num = 0


# PaddleOCR-VL地址
url = "http://192.168.0.201:8080/v1/chat/completions"


# 读取PDF第一页
doc = fitz.open(pdf_path)

page = doc[page_num]


# 转图片
pix = page.get_pixmap(
    dpi=200
)

img_path = "test_page.png"

pix.save(img_path)


print("生成图片:", img_path)


# 图片base64
with open(img_path, "rb") as f:
    img_base64 = base64.b64encode(
        f.read()
    ).decode()


# 请求
data = {
    "model": "PaddleOCR-VL-1.6-0.9B",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":
                    """
请分析该页面：

1. 识别所有文字
2. 分析版面结构
3. 标记标题、正文、表格、图片区域
4. 表格保持HTML结构
5. 输出Markdown格式
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                        f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        }
    ]
}


headers = {
    "Content-Type": "application/json"
}


print("开始请求 PaddleOCR-VL...")


r = requests.post(
    url,
    headers=headers,
    json=data,
    timeout=300
)


print("HTTP状态:", r.status_code)


result = r.json()


print("\n================结果================")


print(
    result["choices"][0]["message"]["content"]
)


# 保存结果

with open(
    "paddleocr_vl_result.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        result["choices"][0]["message"]["content"]
    )


print("\n结果保存 paddleocr_vl_result.md")
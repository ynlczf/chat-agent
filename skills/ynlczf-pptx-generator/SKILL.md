# Skill Name: ynlczf-pptx-generator
## 简介
超轻量化PPT生成Skill，仅依赖python-pptx，无重型库、无动画/渐变/复杂图表，固定极简页面结构，直接输出可编辑`.pptx`文件，适合快速汇报、课堂分享、短演示。
## 触发指令
/pptx 主题:xxx 页数:x 要点:xxx,xxx,xxx
自然语言触发：帮我生成极简PPT、做一份轻量化汇报PPT、导出pptx简报
## 硬性轻量化约束（强制遵守）
1. 仅导入`python-pptx`，禁止pandas/matplotlib/图像处理库，降低算力占用
2. 页面总数≤8页，自动压缩冗余内容，拒绝超长文本
3. 仅使用PPT内置2种基础版式：标题封面、标题+文本要点，不自定义母版/复杂布局
4. 每页要点≤3条，单页文字总量≤60字，无大段段落
5. 禁用阴影、渐变、3D、动画、复杂配色，默认系统简约黑白样式
6. 仅支持极简柱状图（最多4组数据），不混合多图表
7. 输出文件固定：`light_presentation.pptx`，返回本地文件路径
8. 不加载外部图片/字体，纯原生文本+基础图表，文件体积极小
## 标准页面结构（固定逻辑，减少动态计算）
P1 封面：主标题 + 副标题（汇报人/场景）
P2 概述：1段核心目标短句
P3-PN 正文：每页1个模块，3条以内短句要点
倒数第2页（可选）：极简数据柱状图（用户提供数据才生成）
最后1页：总结+下一步规划
## 输入参数规范
用户输入仅接受3类轻量化参数，多余信息自动过滤：
1. topic：PPT主题（必填）
2. page_num：总页数（可选，默认5）
3. points：逗号分隔核心要点列表（必填）
4. data（可选）：数据数组，仅用于生成1张数据页
## 执行流程
1. 解析用户输入，提取主题、页数、要点、数据
2. 初始化空白Presentation对象
3. 循环按固定版式生成幻灯片，填充精简文本
4. 若传入data，追加1页极简柱状图表
5. 保存为`light_presentation.pptx`
6. 返回：PPT生成完成，文件路径=./light_presentation.pptx
## 禁止行为
- 生成超过8页幻灯片
- 添加图片、自定义字体、复杂配色、动画切换
- 生成饼图/折线图混合、多图表页面
- 输出JSON/Markdown提纲替代pptx文件
- 加载本地资源、联网素材、复杂SVG图形
## 配套极简执行代码（skill绑定脚本 scripts/generate_pptx.py）
```python
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

def generate_light_pptx(topic, points, page_num=5, data=None):
    prs = Presentation()
    # P1封面
    cover = prs.slides.add_slide(prs.slide_layouts[0])
    cover.shapes.title.text = topic
    cover.placeholders[1].text = "极简轻量化简报"

    # P2概述
    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "内容概述"
    slide2.placeholders[1].text = f"本次简报核心：{topic}"

    # 正文分页
    max_body_pages = page_num - 2
    for idx, point in enumerate(points[:max_body_pages]):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"模块{idx+1}"
        slide.placeholders[1].text = point

    # 数据页（可选）
    if data and len(data)>=2:
        chart_slide = prs.slides.add_slide(prs.slide_layouts[1])
        chart_slide.shapes.title.text = "核心数据"
        chart_data = CategoryChartData()
        chart_data.categories = [str(i) for i in range(len(data))]
        chart_data.add_series("数值", data)
        chart_slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, 0, 2000000, 6000000, 4000000, chart_data)

    # 总结页
    end_slide = prs.slides.add_slide(prs.slide_layouts[1])
    end_slide.shapes.title.text = "总结与计划"
    end_slide.placeholders[1].text = "1. 完成本次核心内容\n2. 后续持续优化落地"

    save_path = "light_presentation.pptx"
    prs.save(save_path)
    return f"PPT已生成：{save_path}"
"""Generate a reproducible 14k target-domain extension for router training.

The extension deliberately balances three distributions absent or sparse in
LMSYS Arena: generic long-horizon agent work, short Honor phone commands, and
Honor long-horizon agent work.  ``curated_binary_label`` is an auditable seed
label (0=device, 1=cloud), not a replacement for the later real tier label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TOTAL = 14_000
GROUP_SIZES = {
    "generic_long_horizon": 4_666,
    "honor_simple_device": 4_667,
    "honor_long_horizon": 4_667,
}

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "西安", "重庆"]
DOMAINS = ["产品发布", "客户交付", "课程学习", "家庭出行", "求职准备", "健身计划", "毕业设计", "团队协作", "预算管理", "社区活动"]
APPS = ["微信", "日历", "备忘录", "地图", "邮箱", "相册", "文件管理", "荣耀笔记", "健康", "智慧生活"]
DEVICES = ["荣耀手机", "荣耀平板", "荣耀笔记本", "荣耀智慧屏", "荣耀手表"]
TIMES = ["明天早上", "今天下班后", "周五下午", "下周一", "本周末", "晚上八点", "午休时", "通勤路上", "会议开始前", "睡前"]
CONTACTS = ["妈妈", "同事小李", "项目组", "客户王经理", "班主任", "健身教练", "室友", "家人", "导师", "行政同事"]
FOLLOW_UPS = [
    "请保留执行记录。", "关键变更要提醒我。", "先展示结果再继续。", "保持其他设置不变。",
    "如果条件不满足就告诉我原因。", "完成后给我一条简短确认。", "优先使用本机已有信息。", "不要重复创建相同事项。",
    "需要确认的步骤先问我。", "请标记可能的风险。", "结果按优先级展示。", "把未完成项单独列出。",
    "执行前检查是否有冲突。", "如果失败请给替代办法。", "请用简洁中文回复。", "保留我可以撤销的选项。",
    "按时间顺序整理。", "完成后同步更新状态。", "只处理这一次请求。", "先检查当前状态。",
]

GENERIC_LONG_TEMPLATES: Sequence[str] = (
    "围绕{domain}做一个{duration}的完整计划：先汇总现有资料，拆分至少三项有依赖的任务，比较两套方案并说明取舍；执行中若预算或时间变化，更新计划、风险清单和提醒。",
    "把我在{app_a}、{app_b}和{app_c}里的{domain}记录合并去重，找出冲突和缺失信息，按优先级排出{duration}执行表，并为每个关键节点准备失败备选方案。",
    "根据{city}的天气、交通、预算和我的日程，规划{duration}{domain}安排；先给两个候选方案，等我选择后再生成详细清单、预订顺序和变更应对策略。",
    "分析过去{duration}的{domain}数据，定位主要问题，设计三轮改进实验；每轮给出指标、负责人、截止日期和根据结果继续或回退的条件。",
    "为{domain}制作从调研到复盘的工作流：收集资料、归类证据、生成初稿、让不同角色评审、根据反馈迭代，并最后输出决策记录和待办。",
    "我需要处理{domain}的突发变更：先评估对日程、成本和人员的影响，再安排补救步骤；如果关键资源不可用，自动切换到备用方案并通知相关人。",
    "对比三份{domain}方案，按长期成本、风险、可执行性和用户体验打分；给推荐结论、反对理由、实施里程碑和一个月后的复盘指标。",
    "把{domain}拆成连续任务图，标出前置依赖、可并行部分和决策分支；根据我每天可用时间生成{duration}计划，并在延期时重排后续任务。",
)

HONOR_SIMPLE_TEMPLATES: Sequence[str] = (
    "YOYO，{time}提醒我{action}。",
    "在 MagicOS 上{action}。",
    "帮我{action}。",
    "用荣耀手机{action}。",
    "在{device}上{action}。",
    "打开{app_a}并{action}。",
    "把{setting}设置为{value}。",
    "用 Magic 任意门{action}。",
)
SIMPLE_ACTIONS = [
    "打开手电筒", "开启省电模式", "连接蓝牙耳机", "打开相机", "拨打电话给{contact}",
    "导航到公司", "查询今天的天气", "播放本地音乐", "截取长截图", "开启隐私通话",
    "打开多屏协同", "投屏到电视", "扫描这张文档", "旋转这张照片", "把照片裁成正方形",
    "打开健康码", "静音一个小时", "新建一条备忘录", "打开日历", "关闭客厅灯光",
]
SETTINGS = ["屏幕亮度", "铃声音量", "字体大小", "刷新率", "闹钟音量", "护眼模式", "锁屏小组件", "蓝牙"]
VALUES = ["自动", "中等", "最高", "最低", "开启", "关闭", "深色模式", "标准模式"]

HONOR_LONG_TEMPLATES: Sequence[str] = (
    "YOYO，读取{app_a}和{app_b}中的{domain}信息，先提取时间、地点和待办，再检查日历冲突；给两套方案，我确认后分别创建提醒并同步到{device}。",
    "在荣耀多屏协同里汇总{device}和{device_b}的{domain}文件，分类去重、识别版本冲突，生成进度表；缺文件时列清单并提醒负责人补齐。",
    "根据荣耀相册里{duration}的照片和视频，先筛选素材和分析主题，再设计短视频分镜、旁白、封面和发布计划；我选定风格后继续生成编辑步骤。",
    "为{contact}设计 MagicOS 的{duration}辅助使用方案，依次设置字体、语音、紧急联系人、防诈骗和提醒；每周根据反馈调整并保留回退设置。",
    "结合{app_a}的日程、{app_b}的消息和{app_c}的路线，为{city}的{domain}安排连续流程；遇到延误或冲突时重算并通知我。",
    "设计荣耀智慧生活的{domain}自动化：按工作日、天气和是否有人在家判断，依次控制设备；设备失败时记录原因、执行备用动作并发送告警。",
    "分析荣耀手机过去{duration}的耗电、存储、信号和应用使用记录，诊断问题后比较三套优化策略；按风险排序执行，并给每一步的回退方法和两周复盘指标。",
    "用 Magic 任意门处理{domain}海报或截图：识别关键信息、查询路线和日程空档、比较可选方案；确认后创建日历、导航和跨设备待办。",
)


def _format(template: str, index: int) -> str:
    action = SIMPLE_ACTIONS[(index // 10_000) % len(SIMPLE_ACTIONS)].format(
        contact=CONTACTS[(index // 100_000) % len(CONTACTS)]
    )
    return template.format(
        city=CITIES[index % len(CITIES)],
        domain=DOMAINS[(index // 10) % len(DOMAINS)],
        duration=["一周", "两周", "一个月", "两个月", "三天"][(index // 100) % 5],
        app_a=APPS[(index // 500) % len(APPS)],
        app_b=APPS[(index // 5_000) % len(APPS)],
        app_c=APPS[(index // 50_000) % len(APPS)],
        device=DEVICES[(index // 500_000) % len(DEVICES)],
        device_b=DEVICES[(index // 2_500_000) % len(DEVICES)],
        time=TIMES[(index // 100_000) % len(TIMES)],
        contact=CONTACTS[(index // 1_000_000) % len(CONTACTS)],
        action=action,
        setting=SETTINGS[index % len(SETTINGS)],
        value=VALUES[index % len(VALUES)],
    )


def _iter_group(group: str, size: int, templates: Sequence[str], label: int) -> Iterable[dict]:
    for index in range(size):
        template_index = index % len(templates)
        # Generate slots from the per-template counter so each template receives
        # a broad Cartesian coverage instead of repeating every few examples.
        variant_index = index // len(templates)
        text = _format(templates[template_index], variant_index)
        text = f"{text.rstrip('。')}，{FOLLOW_UPS[(variant_index // 50) % len(FOLLOW_UPS)]}"
        # A real mobile request often carries a local clock and task-specific
        # constraints. They make generated examples distinct without changing
        # the routing class: short commands remain bounded local actions.
        day = 1 + (index // 1_440) % 28
        hour = (8 + index // 60) % 24
        minute = index % 60
        if group == "honor_simple_device":
            text = f"设备本地时间 2026-08-{day:02d} {hour:02d}:{minute:02d}，{text}"
        elif group == "generic_long_horizon":
            text = (
                f"设备本地时间 2026-08-{day:02d} {hour:02d}:{minute:02d}，{text}"
                f"本次涉及 {2 + index % 17} 份材料、{3 + (index // 17) % 12} 名参与者，"
                f"预算上限 {1000 + (index // 204) * 200} 元。"
            )
        else:
            text = (
                f"设备本地时间 2026-08-{day:02d} {hour:02d}:{minute:02d}，{text}"
                f"本次关联 {1 + index % 9} 台设备和 {2 + (index // 9) % 15} 份资料。"
            )
        yield {
            "id": f"extension-{group}-{index + 1:05d}",
            "text": text,
            "metadata": {
                "source": "generated_router_extension_v1",
                "extension_group": group,
                "curated_binary_label": label,
                "curated_binary_label_schema": "0=device_local_qwen,1=cloud_deepseek",
                "expected_tier": "device" if label == 0 else "cloud",
                "long_horizon": label == 1,
                "brand_context": "MagicOS/YOYO/Honor mobile ecosystem" if group != "generic_long_horizon" else "general_agent_task",
                "generation_index": index + 1,
            },
        }


def iter_samples() -> Iterable[dict]:
    yield from _iter_group("generic_long_horizon", GROUP_SIZES["generic_long_horizon"], GENERIC_LONG_TEMPLATES, 1)
    yield from _iter_group("honor_simple_device", GROUP_SIZES["honor_simple_device"], HONOR_SIMPLE_TEMPLATES, 0)
    yield from _iter_group("honor_long_horizon", GROUP_SIZES["honor_long_horizon"], HONOR_LONG_TEMPLATES, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/router_learning/router_extension_14k.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_samples())
    assert len(rows) == TOTAL
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(output), "examples": len(rows), "groups": dict(Counter(row["metadata"]["extension_group"] for row in rows)), "labels": dict(Counter(row["metadata"]["curated_binary_label"] for row in rows))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

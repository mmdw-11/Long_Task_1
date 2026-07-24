"""Generate audited Honor mobile prompts for binary device/cloud routing.

Label 0 is a short, local and privacy-preserving command suitable for a
device model. Label 1 is a cloud candidate: it needs long-context reasoning,
at least three dependent actions, online comparison, or multi-stage planning.
The prompts describe intended behaviour only; no real device operation occurs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Each tuple is (prompt, label, routing reason).  They are deliberately written
# as short, natural phone utterances rather than polished benchmark questions.
SAMPLES: Dict[str, List[Tuple[str, int, str]]] = {
    "yoyo_system": [
        ("YOYO，明天早上 7 点叫我起床。", 0, "single offline alarm action"),
        ("打开手电筒。", 0, "single system toggle"),
        ("把手机静音两个小时。", 0, "single local setting"),
        ("连接我的蓝牙耳机。", 0, "single nearby-device action"),
        ("打开微信。", 0, "single app launch"),
        ("根据我下周的日历、目的地天气和会议安排，列出每天穿什么、带什么，发现冲突就给两套出差方案，再把确认项分天提醒我。", 1, "long-horizon calendar reasoning with dependent reminders"),
        ("读取微信里最近的快递通知，提取取件码和到站时间；如果明天到站就建提醒，否则继续关注并在到站后归档。", 1, "cross-app monitoring with conditional branches"),
        ("找出我上个月在 YOYO 记忆里记过的餐厅，按离公司距离、营业时间和人均预算排序，生成周末导航清单。", 1, "memory retrieval, multi-constraint ranking and planning"),
        ("把今天收到的消息按紧急程度分三类，先给我拟回复建议；我确认后再逐条发送并把待办写进日历。", 1, "multi-stage agent workflow requiring confirmation"),
        ("用 Magic 任意门识别这张活动海报，提取时间地点，检查日历冲突，给可选路线并创建提醒。", 1, "image extraction plus schedule and route dependencies"),
    ],
    "mobile_office_learning": [
        ("把这张纸质笔记扫描成清晰 PDF。", 0, "single on-device scan enhancement"),
        ("把这句英文翻译成中文：See you tomorrow.", 0, "short translation"),
        ("开始录音并实时转写。", 0, "single streaming transcription action"),
        ("提取这张图片里的电话号码。", 0, "short OCR extraction"),
        ("把这份文档旋转到正方向。", 0, "single document edit"),
        ("把两个小时的会议录音按发言人分段，提炼决策、待办、责任人和截止日期，找出矛盾项后生成可编辑纪要。", 1, "long audio understanding and structured reconciliation"),
        ("阅读这份 60 页课程 PDF，按章节总结重点，建立知识点关系，再出 10 道由易到难的练习题和答案。", 1, "long-document analysis with chained generation"),
        ("汇总手机、平板里的三份项目周报，去重后按项目列出进展、风险和下周待办，并标记互相冲突的截止日期。", 1, "cross-device document merge and conflict analysis"),
        ("根据我的简历、目标岗位和三份 JD，先找能力缺口，再改三版简历并写不同语气的投递邮件。", 1, "multi-document comparison and iterative writing"),
        ("帮我调试这段报错代码：先定位根因，给最小修复方案，补单元测试，再说明上线前要检查什么。", 1, "dependent code reasoning and verification plan"),
    ],
    "honor_imaging": [
        ("把这张照片裁成 1:1。", 0, "single local image edit"),
        ("删除照片右下角的小杂物。", 0, "bounded image retouching"),
        ("把这张照片亮一点。", 0, "single local adjustment"),
        ("把相机切到人像模式。", 0, "single camera setting"),
        ("把这张照片旋转 90 度。", 0, "single local transform"),
        ("根据这张人像照片生成春日胶片外景方案：先分析人物和光线，再给三种背景与色调，选定后生成修图步骤和朋友圈文案。", 1, "image understanding followed by iterative creative planning"),
        ("筛选相册里所有美食照片，按清晰度和菜品分类，统一给出调色参数，并为每组写一条不同风格的发布文案。", 1, "batch media analysis and multi-output generation"),
        ("用旅行照片和视频做 60 秒短片：先挑素材，设计故事线和分镜，再写旁白、字幕、封面标题和发布标签。", 1, "multi-stage creative pipeline"),
        ("比较三套婚礼跟拍照片的色彩和构图，给统一修图风格，列出每套需修的重点并生成交付检查单。", 1, "multi-set visual comparison and planning"),
        ("根据我给的产品图、卖点和目标用户，连续迭代三版短视频脚本；每版都要说明为何比上一版更适合投放。", 1, "iterative multimodal marketing workflow"),
    ],
    "daily_life_travel": [
        ("导航到公司。", 0, "single navigation action"),
        ("查一下下一班地铁。", 0, "short live query"),
        ("提醒我下班去取快递。", 0, "single reminder"),
        ("打开客厅空调。", 0, "single smart-home command"),
        ("查今天北京的天气。", 0, "short weather query"),
        ("规划周末两天从北京出发的自驾游：结合预算、儿童同行、天气和拥堵，给路线、景点、餐厅、住宿和下雨备选方案。", 1, "multi-constraint itinerary with contingency planning"),
        ("预算 300 元，筛选周五下午出发的高铁和车站附近酒店；比较总耗时与价格，订前把最优方案发我确认。", 1, "online search, optimization and confirmation step"),
        ("比较附近三家适合家庭聚餐的餐厅，综合评价、排队时间、儿童座椅和菜品，给点菜清单和两套备选。", 1, "multi-source decision analysis"),
        ("按我本月消费记录制定下月餐饮、交通和娱乐预算；找出超支原因，给每周额度和可执行的省钱提醒。", 1, "longitudinal personal-data analysis and plan"),
        ("根据下周五天的会议地点规划每天通勤路线，同时安排午餐和充电时间；有日程变更时重新计算并通知我。", 1, "persistent, conditional multi-day scheduling"),
    ],
    "cross_device_smart_life": [
        ("把手机屏幕投到电视上。", 0, "single device handoff"),
        ("在平板上打开这份文档。", 0, "single cross-device open action"),
        ("关掉客厅的灯。", 0, "single IoT command"),
        ("把这个文件传到我的电脑。", 0, "single nearby transfer"),
        ("开启多屏协同。", 0, "single connectivity setting"),
        ("设计荣耀智慧生活的离家模式：检测人是否离开后依次关闭电器、检查门锁、开启摄像头；任一设备失败时给我告警和补救建议。", 1, "conditional multi-device automation"),
        ("整理平板、电脑和手机里的工作文档，按项目分类、去重并生成目录；把缺失版本和待确认文件单独列出来。", 1, "cross-device batch processing and reconciliation"),
        ("为折叠屏设计一套通勤工作流：外屏处理消息，展开屏阅读资料，到公司自动把未完成任务同步电脑，并记录失败步骤。", 1, "stateful workflow across devices and contexts"),
        ("把手机备忘录和荣耀平板笔记里的项目记录合并，识别重复和冲突，输出每个项目的进度、风险、负责人和下一步。", 1, "long-context cross-source synthesis"),
        ("建立早间、离家、归家三套自动化场景，分别考虑工作日/周末、天气和家庭成员在家状态，并给出测试步骤。", 1, "branching automation design with validation"),
    ],
    "privacy_and_system_care": [
        ("开启隐私通话。", 0, "single local privacy toggle"),
        ("打开省电模式。", 0, "single local setting"),
        ("截一张长图。", 0, "single system operation"),
        ("把锁屏小组件换成天气。", 0, "single personalization action"),
        ("开启指关节双击截图。", 0, "single MagicOS gesture setting"),
        ("根据我一周的耗电、应用使用和信号数据，找出续航问题，比较三套刷新率和后台策略，并生成可回退的优化方案。", 1, "time-series analysis and multi-option optimization"),
        ("分析最近一个月的卡顿日志、存储和应用权限，按影响程度给分步排障；每一步说明风险、预期效果和如何撤销。", 1, "multi-source diagnosis with safe rollback plan"),
        ("盘点手机里的照片、聊天备份和应用权限，识别可能的隐私暴露；先给脱敏和本地处理方案，再列出确需上云的最小数据。", 1, "sensitive-data policy reasoning and staged plan"),
        ("为家里老人定制一个月的手机辅助使用方案：字体、语音、紧急联系人、防诈骗和远程协助，分周教学并预留反馈调整。", 1, "long-horizon personalized plan"),
        ("根据我的使用习惯设计工作、睡眠和专注三种模式；设置通知、亮度、应用限制和自动切换条件，再给两周复盘指标。", 1, "personalized conditional policy design"),
    ],
    "long_horizon_agent": [
        ("周五下午 3 点提醒我交报告。", 0, "single reminder"),
        ("把明天的会议改到下午。", 0, "single calendar edit"),
        ("播放我下载的轻音乐。", 0, "single local media command"),
        ("给妈妈打电话。", 0, "single communication action"),
        ("打开健康码。", 0, "single app action"),
        ("持续关注微信和邮件里的面试通知，提取公司、岗位和时间，创建日历，提前一天提醒我，并根据岗位生成面试题清单。", 1, "persistent monitoring with extraction, scheduling and generation"),
        ("为两个月备考制定学习表：先根据我的空闲日历拆分科目，再每周根据完成情况调整，并在薄弱章节生成练习和复盘。", 1, "multi-week adaptive planning"),
        ("规划下周跨城出差全流程：交通、会议、客户沟通、行李、本地就餐和报销；航班变动时给替代方案并更新日程。", 1, "long itinerary with dynamic contingency handling"),
        ("把三份会议纪要、聊天记录和任务看板合并，找出依赖关系和冲突，排出两周执行计划，并在关键节点提醒负责人。", 1, "cross-source dependency planning"),
        ("帮我策划周末露营：先给两套方案并比较成本和风险，我选定后再写招募文案、物品清单、分工、天气预案和现场流程。", 1, "multi-phase plan with user decision branch"),
    ],
}


def iter_samples() -> Iterable[dict]:
    for scenario, rows in SAMPLES.items():
        for index, (text, label, reason) in enumerate(rows):
            yield {
                "id": f"honor-{scenario}-{index:02d}",
                "text": text,
                "label": label,
                "metadata": {
                    "source": "curated_honor_mobile_router",
                    "scenario": scenario,
                    "route_label_schema": "0=device_local_qwen,1=cloud_deepseek",
                    "expected_tier": "device" if label == 0 else "cloud",
                    "routing_reason": reason,
                    "long_horizon": label == 1,
                    "brand_context": "MagicOS/YOYO/Honor mobile ecosystem",
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/router_learning/honor_phone_router_dataset.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_samples())
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    counts = Counter(row["label"] for row in rows)
    print(json.dumps({"output": str(output), "examples": len(rows), "label_counts": dict(counts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""旅行规划示例的工具集（AgentScope 2.x 工具写法）。

每个工具都是一个普通 Python 函数，返回 :class:`agentscope.tool.ToolResponse`：

- 名称取自函数名，描述取自 docstring，入参 schema 由类型注解自动推断。
- 通过 :class:`agentscope.tool.FunctionTool` 包装，再放进 :class:`~agentscope.tool.Toolkit`，
  即可挂载到 Agent 上供 ReAct 循环自主调用。

包含的工具：

- ``estimate_budget``    —— 基于人数/天数粗略估算预算明细。
- ``split_expense``      —— 计算总费用的人均分摊（及可选每日均摊）。
- ``sum_expenses``       —— 汇总多项开销并给出总额。
- ``web_search``         —— 使用 DuckDuckGo 搜索引擎获取结果。
- ``browse_webpage``     —— 使用 Playwright 打开网页并抽取可见文本。

搜索工具依赖 ``ddgs`` 库（``pip install ddgs``）。
浏览网页工具依赖 Playwright（``pip install playwright && python -m playwright install chromium``）。
若依赖缺失，工具会返回 ``state=ERROR`` 的 ToolResponse，
Agent 可据此改用自身知识继续规划，而不会中断整条流水线。
"""

from __future__ import annotations

import json
import re
from typing import Callable

from agentscope.message import TextBlock
from agentscope.message import ToolResultState
from agentscope.tool import FunctionTool, ToolResponse, Toolkit


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _text_response(text: str, *, error: bool = False, **metadata) -> ToolResponse:
    """构造一个纯文本 ToolResponse。"""
    return ToolResponse(
        content=[TextBlock(type="text", text=text)],
        state=ToolResultState.ERROR if error else ToolResultState.SUCCESS,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# 工具 1：估算预算
# ---------------------------------------------------------------------------
def estimate_budget(days: int, people: int) -> ToolResponse:
    """基于出行天数与人数粗略估算旅行预算（人民币，元）。

    Args:
        days: 出行总天数。
        people: 出行总人数。
    """
    people = max(1, people)
    days = max(1, days)
    transport = 800 * people
    hotel = 420 * max(0, days - 1)
    food = 160 * days * people
    local = 70 * days * people
    tickets = 90 * days * people
    buffer = 500
    breakdown = {
        "transport": transport,
        "hotel": hotel,
        "food": food,
        "local_transport": local,
        "tickets": tickets,
        "buffer": buffer,
        "total": transport + hotel + food + local + tickets + buffer,
    }
    return _text_response(
        json.dumps(breakdown, ensure_ascii=False, indent=2), **breakdown
    )


# ---------------------------------------------------------------------------
# 工具 2：人均分摊
# ---------------------------------------------------------------------------
def split_expense(total_amount: float, people: int, days: int = 0) -> ToolResponse:
    """计算总费用的人均分摊，可选给出每人每日均摊。

    Args:
        total_amount: 需要分摊的总金额（元）。
        people: 分摊人数。
        days: 出行天数；传入 >0 时额外给出每人每日均摊。
    """
    people = max(1, people)
    per_person = round(total_amount / people, 2)
    result = {
        "total_amount": total_amount,
        "people": people,
        "per_person": per_person,
    }
    if days > 0:
        result["days"] = days
        result["per_person_per_day"] = round(per_person / days, 2)
    return _text_response(
        json.dumps(result, ensure_ascii=False, indent=2), **result
    )


# ---------------------------------------------------------------------------
# 工具 3：汇总开销
# ---------------------------------------------------------------------------
def sum_expenses(amounts: list[float]) -> ToolResponse:
    """汇总多项开销金额，返回总额、项数与平均值。

    Args:
        amounts: 各项开销金额列表（元），例如 [1600, 1680, 800]。
    """
    valid = [float(x) for x in amounts]
    total = round(sum(valid), 2)
    count = len(valid)
    result = {
        "total": total,
        "count": count,
        "average": round(total / count, 2) if count else 0.0,
    }
    return _text_response(
        json.dumps(result, ensure_ascii=False, indent=2), **result
    )


# ---------------------------------------------------------------------------
# 工具 4：Web 搜索（DuckDuckGo）
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> ToolResponse:
    """使用 DuckDuckGo 搜索引擎检索网络信息，返回标题/链接/摘要。

    Args:
        query: 搜索关键词，例如"成都 5月 天气 旅游"。
        max_results: 返回结果数量上限，默认 5。
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return _text_response(
            "ddgs 未安装，无法执行 web_search。请先运行：pip install ddgs",
            error=True,
        )

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001 - 网络/限流异常统一降级
        return _text_response(f"web_search 执行失败：{exc}", error=True)

    if not results:
        return _text_response(f'未检索到与“{query}”相关的结果。', results=[])

    lines = [f"检索关键词：{query}", ""]
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. {item['title']}\n   {item['href']}\n   {item['body']}")
    return _text_response("\n".join(lines), results=results)


# 用于伪装成常见桌面浏览器，降低被网页判定为爬虫的概率。
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# 工具 5：浏览网页（Playwright）
# ---------------------------------------------------------------------------
async def browse_webpage(url: str, max_chars: int = 4000) -> ToolResponse:
    """使用真实浏览器（Playwright）打开网页并抽取可见正文文本。

    Args:
        url: 要访问的网页完整地址（需以 http/https 开头）。
        max_chars: 返回正文的最大字符数，超出部分会被截断，默认 4000。
    """
    if not url.startswith(("http://", "https://")):
        return _text_response(f"非法 URL（需以 http/https 开头）：{url}", error=True)

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return _text_response(
            "Playwright 未安装，无法执行 browse_webpage。请先运行："
            "pip install playwright && python -m playwright install chromium",
            error=True,
        )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=_USER_AGENT)
            await page.goto(url, timeout=30000, wait_until="load")
            title = await page.title()
            # 等待页面稳定（防止跳转/重定向销毁执行上下文）
            await page.wait_for_timeout(2000)
            try:
                body_text = await page.evaluate(
                    "() => (document.body && document.body.innerText) || ''"
                )
            except Exception:
                # 页面已跳转/销毁，尝试取新页面内容
                title = await page.title()
                body_text = await page.evaluate(
                    "() => (document.body && document.body.innerText) || ''"
                )
            await browser.close()
    except Exception as exc:  # noqa: BLE001 - 浏览器/网络异常统一降级
        return _text_response(f"browse_webpage 执行失败：{exc}", error=True)

    text = re.sub(r"\n{3,}", "\n\n", (body_text or "").strip())
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n…（内容已截断）"
    header = f"标题：{title}\n链接：{url}\n"
    return _text_response(header + "\n" + text, title=title, url=url, truncated=truncated)


# ---------------------------------------------------------------------------
# 工具注册表与 Toolkit 工厂
# ---------------------------------------------------------------------------
TOOL_REGISTRY: dict[str, Callable] = {
    "estimate_budget": estimate_budget,
    "split_expense": split_expense,
    "sum_expenses": sum_expenses,
    "web_search": web_search,
    "browse_webpage": browse_webpage,
}


def build_toolkit(names: list[str]) -> Toolkit | None:
    """根据工具名列表构造一个 Toolkit；名称为空时返回 None。

    Args:
        names: 工具名列表，取值须在 :data:`TOOL_REGISTRY` 中。
    """
    if not names:
        return None
    tools = [FunctionTool(TOOL_REGISTRY[name]) for name in names if name in TOOL_REGISTRY]
    if not tools:
        return None
    return Toolkit(tools=tools)

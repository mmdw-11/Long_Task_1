"""Request-local runtime telemetry; never emit prompts or provider reasoning."""
from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
import threading
from types import SimpleNamespace

sink = contextvars.ContextVar("runtime_event_sink", default=None)
node_name = contextvars.ContextVar("runtime_node", default="")
cancel_signal = contextvars.ContextVar("runtime_cancel", default=None)


def checkpoint():
    signal = cancel_signal.get()
    if signal is not None and signal.is_set():
        raise RuntimeError("运行已取消；不会开始后续操作")


def emit(kind, **payload):
    callback = sink.get()
    if callback:
        if kind not in {"tool_finished", "model_failed"}:
            checkpoint()
        callback({"type": kind, "node": node_name.get(), **payload})


async def live_events(iterator, cancel_check=None):
    """Merge worker telemetry and graph events without blocking the SSE reader."""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    sentinel = object()
    canceled = threading.Event()
    loop_thread = threading.get_ident()

    def publish(event):
        if threading.get_ident() == loop_thread:
            queue.put_nowait(event)
        else:
            loop.call_soon_threadsafe(queue.put_nowait, event)

    async def produce():
        token = sink.set(publish)
        cancel_token = cancel_signal.set(canceled)
        try:
            async for event in iterator:
                await queue.put(event)
        except BaseException as exc:
            await queue.put(exc)
        finally:
            sink.reset(token)
            cancel_signal.reset(cancel_token)
            await queue.put(sentinel)

    task = asyncio.create_task(produce())
    async def watch_cancel():
        while not task.done():
            if cancel_check and cancel_check():
                canceled.set()
                return
            await asyncio.sleep(.1)

    watcher = asyncio.create_task(watch_cancel())
    try:
        while True:
            event = await queue.get()
            if event is sentinel:
                break
            if isinstance(event, BaseException):
                raise event
            yield event
    finally:
        canceled.set()
        if not task.done():
            task.cancel()
        watcher.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)


class VisibleText:
    """Suppress tagged reasoning, including tags split between network chunks."""
    def __init__(self):
        self.buffer = ""
        self.hidden = None

    def feed(self, text, final=False):
        self.buffer += text
        visible = []
        while self.buffer:
            tags = [f"</{self.hidden}>"] if self.hidden else ["<think>", "<analysis>"]
            lowered = self.buffer.lower()
            matches = [(lowered.find(tag), tag) for tag in tags if tag in lowered]
            if matches:
                index, tag = min(matches)
                if not self.hidden:
                    visible.append(self.buffer[:index])
                self.buffer = self.buffer[index + len(tag):]
                self.hidden = None if self.hidden else tag[1:-1]
                continue
            hold = 0 if final else max([n for tag in tags for n in range(1, len(tag)) if lowered.endswith(tag[:n])] or [0])
            if not self.hidden:
                visible.append(self.buffer[:-hold] if hold else self.buffer)
            self.buffer = self.buffer[-hold:] if hold else ""
            break
        return "".join(visible)


def chat_completion(create, **kwargs):
    """Keep non-streaming callers compatible; collect fragmented tool arguments.

    Reasoning is retained privately only for provider tool-call continuation.
    Each generation has a separate id, so retries never concatenate answers.
    No fallback request is made after a partially consumed response.
    """
    if sink.get() is None:
        return create(**kwargs)
    checkpoint()
    generation = uuid.uuid4().hex
    emit("model_started", generation_id=generation, model=kwargs.get("model"), message="模型正在生成回复")
    chunks, pending, calls, reasoning = [], [], {}, []
    last_flush = time.monotonic()
    stream = None
    visible_text = VisibleText()
    finish_reason = None
    usage = None

    def flush():
        nonlocal last_flush
        if pending:
            emit("answer_delta", generation_id=generation, delta="".join(pending))
            pending.clear()
        last_flush = time.monotonic()

    try:
        try:
            stream = create(**kwargs, stream=True)
        except Exception as exc:
            detail = str(exc).lower()
            unsupported = getattr(exc, "status_code", None) in {400, 422} and "stream" in detail and any(word in detail for word in ("unsupported", "not supported", "not support"))
            if not unsupported:
                raise
            # Only an explicit rejection before receiving any output permits
            # a non-streaming request. Never retry a partially consumed stream.
            emit("answer_mode", message="该模型服务不支持流式输出，将在生成完成后显示正文")
            response = create(**kwargs, stream=False)
            message = response.choices[0].message
            message.content = visible_text.feed(message.content or "", final=True)
            emit("model_finished", generation_id=generation, message="模型已完成非流式生成")
            return response
        for chunk in stream:
            checkpoint()
            usage = getattr(chunk, "usage", None) or usage
            for choice in chunk.choices:
                if getattr(choice, "index", 0) != 0:
                    continue
                delta = choice.delta
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                if getattr(delta, "content", None):
                    text = visible_text.feed(delta.content)
                    chunks.append(text)
                    pending.append(text)
                if getattr(delta, "reasoning_content", None):
                    reasoning.append(delta.reasoning_content)
                for part in getattr(delta, "tool_calls", None) or []:
                    call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                    if part.id:
                        call["id"] = part.id
                    function = getattr(part, "function", None)
                    if function:
                        call["name"] += function.name or ""
                        call["arguments"] += function.arguments or ""
            if time.monotonic() - last_flush >= .1 or sum(map(len, pending)) >= 256:
                flush()
        tail = visible_text.feed("", final=True)
        chunks.append(tail)
        pending.append(tail)
        flush()
        if finish_reason not in {"stop", "tool_calls", "function_call"}:
            raise RuntimeError(f"模型输出未完整结束（{finish_reason or '连接提前关闭'}）")
        emit("model_finished", generation_id=generation, message="模型已选择工具，准备执行" if calls else "本轮模型回复生成完成")
        message = SimpleNamespace(content="".join(chunks), reasoning_content="".join(reasoning), tool_calls=[
            SimpleNamespace(id=value["id"], type="function", function=SimpleNamespace(name=value["name"], arguments=value["arguments"]))
            for _, value in sorted(calls.items())
        ])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
    except Exception:
        flush()
        emit("model_failed", generation_id=generation, message="模型连接失败或输出中断，已保留收到的正文")
        raise
    finally:
        if stream is not None and hasattr(stream, "close"):
            stream.close()

#!/usr/bin/env python3
"""Combined demo: FastRTC phone preview + Open-AutoGLM PhoneAgent.

This is a standalone UI that lets you:
- Stream your Android phone screen to the browser via FastRTC WebRTC (server -> client).
- Run a PhoneAgent task and view streaming logs.

Notes:
- Preview stream uses ADB screenshots (adb exec-out screencap -p). FPS is limited.
- Do NOT hardcode API keys; provide them via the UI/env.

Run:
  D:/workspace/copilot-workspace/.venv/Scripts/python.exe demo_fastrtc_agent_ui.py

If port is in use, set:
  GRADIO_SERVER_PORT=7863
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import sys
from io import BytesIO
from pathlib import Path
from queue import Queue, Empty

import gradio as gr
import numpy as np
from fastrtc import WebRTC
from PIL import Image


_RUNNING_LOCK = threading.Lock()
_RUNNING_PROCS: dict[str, subprocess.Popen] = {}
_LAST_DEVICE_ID: dict[str, str] = {}


def _session_key(request: gr.Request | None) -> str:
    if request is None:
        return "<no-request>"
    return getattr(request, "session_hash", None) or "<no-session>"


def _set_running_proc(key: str, proc: subprocess.Popen) -> None:
    with _RUNNING_LOCK:
        _RUNNING_PROCS[key] = proc


def _set_last_device_id(key: str, device_id: str) -> None:
    with _RUNNING_LOCK:
        if device_id:
            _LAST_DEVICE_ID[key] = device_id


def _get_last_device_id(key: str) -> str:
    with _RUNNING_LOCK:
        return _LAST_DEVICE_ID.get(key, "")


def _get_running_proc(key: str) -> subprocess.Popen | None:
    with _RUNNING_LOCK:
        return _RUNNING_PROCS.get(key)


def _clear_running_proc(key: str) -> None:
    with _RUNNING_LOCK:
        _RUNNING_PROCS.pop(key, None)


def _adb_go_home(device_id: str) -> tuple[bool, str]:
    """Best-effort: send HOME key to return to launcher."""

    _ensure_adb_on_path()
    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]
    cmd += ["shell", "input", "keyevent", "3"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=5)
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return False, err or f"adb exited with code {proc.returncode}"
    except Exception as e:
        return False, str(e)


def _ensure_adb_on_path() -> None:
    if shutil.which("adb") is not None:
        return

    here = Path(__file__).resolve()
    # .../autoGLM/Open-AutoGLM-main/demo_fastrtc_agent_ui.py -> workspace/tools/platform-tools
    candidate = here.parents[2] / "tools" / "platform-tools"
    if candidate.exists():
        os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")


def _list_adb_devices_lines() -> list[str]:
    _ensure_adb_on_path()
    try:
        proc = subprocess.run(["adb", "devices", "-l"], capture_output=True, timeout=5)
    except Exception:
        return []

    text = (proc.stdout or b"").decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("list of devices"):
        lines = lines[1:]
    return lines


def refresh_devices() -> tuple[gr.Dropdown, str]:
    lines = _list_adb_devices_lines()
    if not lines:
        return gr.update(choices=[], value=None), "No devices detected. Check USB debugging and authorization."

    choices: list[str] = []
    preferred: str | None = None

    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        device_id = parts[0]
        status = parts[1] if len(parts) > 1 else ""
        label = f"{device_id} [{status}]"
        choices.append(label)
        if status == "device" and preferred is None:
            preferred = label

    if not choices:
        return gr.update(choices=[], value=None), "No parsable devices."

    return gr.update(choices=choices, value=preferred or choices[0]), f"Detected {len(choices)} device(s)."


def _extract_device_id(choice: str) -> str:
    return (choice or "").split()[0]


def _adb_screencap_png(device_id: str | None, timeout_s: float = 8.0) -> bytes:
    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]
    cmd += ["exec-out", "screencap", "-p"]

    proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"adb exited with code {proc.returncode}")

    if not proc.stdout:
        raise RuntimeError("empty screencap output")

    return proc.stdout


def stream_phone_screen(device_choice: str, device_id_manual: str, fps: int):
    """FastRTC WebRTC receive-only video stream handler.

    Yields frames as numpy arrays.
    """

    _ensure_adb_on_path()

    device_id = (device_id_manual or "").strip() or _extract_device_id(device_choice)
    if not device_id:
        raise gr.Error("Please select or type a device id.")

    fps = int(fps)
    if fps < 1:
        fps = 1
    if fps > 10:
        fps = 10

    interval = 1.0 / float(fps)

    # Send the first frame immediately so users see something fast.
    next_at = 0.0
    while True:
        now = time.time()
        if now < next_at:
            time.sleep(min(0.05, next_at - now))
            continue
        next_at = time.time() + interval

        png = _adb_screencap_png(device_id)
        # FastRTC encodes outgoing frames as bgr24 internally.
        # Provide BGR here to avoid swapped colors.
        img = Image.open(BytesIO(png)).convert("RGB")
        rgb = np.asarray(img)
        bgr = rgb[:, :, ::-1]
        yield bgr


def run_task(
    base_url: str,
    model_name: str,
    api_key: str,
    lang: str,
    max_steps: int,
    verbose: bool,
    device_choice: str,
    device_id_manual: str,
    task: str,
    request: gr.Request,
):
    _ensure_adb_on_path()

    if not task or not task.strip():
        yield "", "Please provide a task."
        return

    device_id = (device_id_manual or "").strip() or _extract_device_id(device_choice)

    key = _session_key(request)
    _set_last_device_id(key, device_id or "")
    existing = _get_running_proc(key)
    if existing is not None and existing.poll() is None:
        yield "", "A task is already running. Click Cancel first."
        return

    runner = Path(__file__).with_name("_phone_agent_task_runner.py")
    if not runner.exists():
        yield "", f"Missing runner script: {runner}"
        return

    cmd = [
        sys.executable,
        str(runner),
        "--base-url",
        (base_url or "").strip(),
        "--model",
        (model_name or "").strip(),
        "--lang",
        (lang or "cn").strip(),
        "--max-steps",
        str(int(max_steps)),
        "--device-id",
        (device_id or ""),
        "--task-stdin",
    ]
    if bool(verbose):
        cmd.append("--verbose")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PHONE_AGENT_API_KEY"] = (api_key or "").strip()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    _set_running_proc(key, proc)

    q: Queue[tuple[str, str]] = Queue()

    def _pump(stream, tag: str):
        try:
            for line in iter(stream.readline, ""):
                q.put((tag, line))
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    try:
        if proc.stdin is not None:
            proc.stdin.write(task)
            proc.stdin.close()
    except Exception:
        pass

    logs_acc = ""
    last_sent = ""
    result_value = ""
    RESULT_PREFIX = "__PHONE_AGENT_RESULT__="

    yield "", "Starting..."

    try:
        while True:
            drained = False
            while True:
                try:
                    tag, line = q.get_nowait()
                except Empty:
                    break
                drained = True

                if tag == "stdout" and line.startswith(RESULT_PREFIX):
                    payload = line[len(RESULT_PREFIX) :].strip()
                    try:
                        data = json.loads(payload)
                        result_value = str(data.get("result", ""))
                    except Exception:
                        result_value = payload
                    continue

                prefix = "" if tag == "stdout" else "[stderr] "
                logs_acc += prefix + line

            if logs_acc != last_sent:
                last_sent = logs_acc
                yield "", logs_acc

            exited = proc.poll() is not None
            if exited:
                # After the subprocess exits, keep draining any remaining stdout/stderr.
                # Once the queue is empty, we can finish and return the final result.
                if q.empty():
                    break
                time.sleep(0.05)
                continue

            if not drained:
                time.sleep(0.1)

    finally:
        # Ensure process is not left running.
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            pass
        _clear_running_proc(key)

    rc = proc.returncode
    if rc not in (0, None) and not result_value:
        logs_acc += ("\n" if logs_acc else "") + f"[exit] code={rc}"

    yield result_value, logs_acc


def cancel_task(current_logs: str, request: gr.Request):
    """Best-effort terminate the running PhoneAgent subprocess for this session."""

    key = _session_key(request)
    device_id = _get_last_device_id(key)
    proc = _get_running_proc(key)

    cancelled = False
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
            cancelled = True
        except Exception:
            pass
        finally:
            _clear_running_proc(key)
    else:
        _clear_running_proc(key)

    ok, err = _adb_go_home(device_id)
    msg = "[cancel] Task cancelled (subprocess terminated).\n" if cancelled else "[cancel] No running task to cancel.\n"
    if ok:
        msg += "[safety] Sent HOME key (return to launcher).\n"
    else:
        msg += f"[safety] Failed to send HOME: {err}\n"

    return "", (msg + (current_logs or ""))


def main() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _ensure_adb_on_path()

    with gr.Blocks(title="Open-AutoGLM (Phone Preview + Agent)") as demo:
        gr.Markdown(
            "# Open-AutoGLM\n"
            "One UI: phone screen preview (WebRTC) + PhoneAgent task execution."  
        )

        with gr.Accordion("Config", open=False):
            base_url = gr.Textbox(
                label="Base URL",
                value=os.getenv("PHONE_AGENT_BASE_URL", "https://api-inference.modelscope.cn/v1"),
            )
            model_name = gr.Textbox(
                label="Model",
                value=os.getenv("PHONE_AGENT_MODEL", "ZhipuAI/AutoGLM-Phone-9B"),
            )
            api_key = gr.Textbox(
                label="API Key / Token",
                type="password",
                value=os.getenv("PHONE_AGENT_API_KEY", ""),
                placeholder="ms-...",
            )

            with gr.Row():
                lang = gr.Dropdown(label="Language", choices=["cn", "en"], value=os.getenv("PHONE_AGENT_LANG", "cn"))
                max_steps = gr.Slider(label="Max Steps", minimum=1, maximum=200, step=1, value=int(os.getenv("PHONE_AGENT_MAX_STEPS", "100")))
                verbose = gr.Checkbox(label="Verbose", value=True)

        with gr.Accordion("Device (ADB)", open=True):
            with gr.Row():
                refresh_btn = gr.Button("Refresh devices")
                start_preview_btn = gr.Button("Start preview")
                device_status = gr.Markdown("")
            with gr.Row():
                device_choice = gr.Dropdown(label="Detected device", choices=[], value=None, allow_custom_value=True)
                device_id_manual = gr.Textbox(label="Or type device id", value=os.getenv("PHONE_AGENT_DEVICE_ID", ""))
            fps = gr.Slider(label="Preview FPS (ADB)", minimum=1, maximum=10, step=1, value=3)

        with gr.Row():
            with gr.Column(scale=1):
                preview = WebRTC(label="Phone Screen (WebRTC)", mode="receive", modality="video",height=800)
            with gr.Column(scale=1):
                task = gr.Textbox(label="Task", lines=4, placeholder="例如：打开今日头条，搜索‘新能源汽车’")
                with gr.Row():
                    run_btn = gr.Button("Run task")
                    cancel_btn = gr.Button("Cancel")
                result = gr.Textbox(label="Result", lines=4)
                logs = gr.Textbox(label="Logs", lines=18)

        refresh_btn.click(fn=refresh_devices, inputs=None, outputs=[device_choice, device_status])
        demo.load(fn=refresh_devices, inputs=None, outputs=[device_choice, device_status])

        # FastRTC receive-mode requires an explicit trigger.
        preview.stream(
            fn=stream_phone_screen,
            inputs=[device_choice, device_id_manual, fps],
            outputs=[preview],
            time_limit=60 * 60,
            trigger=start_preview_btn.click,
        )

        run_event = run_btn.click(
            fn=run_task,
            inputs=[base_url, model_name, api_key, lang, max_steps, verbose, device_choice, device_id_manual, task],
            outputs=[result, logs],
        )

        # Allow pressing Enter in the task box to run.
        submit_event = task.submit(
            fn=run_task,
            inputs=[base_url, model_name, api_key, lang, max_steps, verbose, device_choice, device_id_manual, task],
            outputs=[result, logs],
        )

        # Cancel a running task execution (requires queue mode).
        cancel_btn.click(
            fn=cancel_task,
            inputs=[logs],
            outputs=[result, logs],
            cancels=[run_event, submit_event],
        )

    port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PHONE_AGENT_GRADIO_PORT", "7863")))
    demo.queue(max_size=16).launch(server_name="127.0.0.1", server_port=port, show_error=True)


if __name__ == "__main__":
    main()

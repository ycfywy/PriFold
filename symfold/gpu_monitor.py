"""GPU 实时监控工具（独立守护进程）。

设计原则
--------
**与训练/评估进程完全解耦**：作为后台 daemon 独立运行，定时采样写 JSONL。
- 训练崩了，监控还在；监控崩了，训练不受影响
- 采样频率独立可控（默认 5s）
- 一份 monitor 可同时监控多个训练任务（每个任务各自的 jsonl 文件）

数据来源
--------
- NVML（pynvml）：全卡视角的显存、SM/Mem 利用率、温度、功率
- /proc/<pid>/status + nvml_get_compute_processes：定位训练 PID 在 GPU 上的占用
- 不依赖 torch（当 daemon 独立跑时）

用法（命令行）
--------------
    # 启动一个后台 daemon，每 5s 采样一次写到 outputs/v2_bprna/gpu_stats.jsonl
    python -m symfold.gpu_monitor daemon \\
        --out symfold/outputs/v2_bprna/gpu_stats.jsonl \\
        --device 0 --interval 5

    # 实时查看
    tail -f symfold/outputs/v2_bprna/gpu_stats.jsonl

    # 表格查看
    python -m symfold.show_gpu_stats symfold/outputs/v2_bprna/gpu_stats.jsonl --tail 20
    python -m symfold.show_gpu_stats symfold/outputs/v2_bprna/gpu_stats.jsonl --summary

run_train.sh 已自动配套启动一个 daemon，无需手动管理。

用法（程序内调用）
------------------
若仍想在训练代码里打"标记点"（标注当前 epoch/step 给离线对齐时间轴用），
仍然可以通过 GpuMonitor.snapshot() 同步取一个快照写到独立 marks.jsonl，
但那不是必需的。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import pynvml  # type: ignore
    _NVML_OK = True
except ImportError:
    _NVML_OK = False


# ============================================================
# Snapshot helpers
# ============================================================

def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


class GpuMonitor:
    """轻量级快照工具。可在训练进程内同步使用，也可在 daemon 内被反复调用。"""

    def __init__(self, device_index: int = 0, target_pid: Optional[int] = None):
        self.device_index = device_index
        self.target_pid = target_pid  # 训练进程 PID（若已知，daemon 可定位它）
        self.handle = None
        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                self.handle = None

    def snapshot(self, phase: str = 'monitor', extra: Optional[dict] = None) -> dict:
        """返回当前 GPU 状态字典。"""
        out: dict = {
            'ts': time.time(),
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase': phase,
        }
        if self.handle is not None:
            mem = _try(lambda: pynvml.nvmlDeviceGetMemoryInfo(self.handle))
            util = _try(lambda: pynvml.nvmlDeviceGetUtilizationRates(self.handle))
            if mem is not None:
                out['nvml_used_mb'] = round(mem.used / 1024 / 1024, 1)
                out['nvml_total_mb'] = round(mem.total / 1024 / 1024, 1)
                out['nvml_free_mb'] = round(mem.free / 1024 / 1024, 1)
            if util is not None:
                out['nvml_util_gpu'] = int(util.gpu)
                out['nvml_util_mem'] = int(util.memory)
            temp = _try(lambda: pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU))
            if temp is not None:
                out['nvml_temp_c'] = int(temp)
            power = _try(lambda: pynvml.nvmlDeviceGetPowerUsage(self.handle))
            if power is not None:
                out['nvml_power_w'] = round(power / 1000.0, 1)
            # 单进程显存（在该 GPU 上跑的所有进程）
            procs = _try(lambda: pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle), default=[])
            if procs is not None:
                proc_list = []
                target_used = None
                for p in procs:
                    pid_used = round(p.usedGpuMemory / 1024 / 1024, 1) \
                        if p.usedGpuMemory is not None else None
                    proc_list.append({'pid': int(p.pid), 'used_mb': pid_used})
                    if self.target_pid is not None and int(p.pid) == self.target_pid:
                        target_used = pid_used
                # PID namespace fallback: NVML 看的是 host PID, 训练 PID 可能是容器 PID,
                # 若没直接命中且 GPU 上只有一个 compute 进程，则把它当 target
                if target_used is None and self.target_pid is not None and len(proc_list) == 1:
                    target_used = proc_list[0]['used_mb']
                # 还可以尝试通过 /proc/<container_pid>/status: NSpid 找到 host PID
                if target_used is None and self.target_pid is not None:
                    host_pid = _container_to_host_pid(self.target_pid)
                    if host_pid is not None:
                        for p in proc_list:
                            if p['pid'] == host_pid:
                                target_used = p['used_mb']
                                out['target_host_pid'] = host_pid
                                break
                out['gpu_procs'] = proc_list
                if target_used is not None:
                    out['target_pid_used_mb'] = target_used
                if self.target_pid is not None:
                    out['target_pid'] = self.target_pid
                    out['target_alive'] = pid_alive(self.target_pid)
        if extra:
            out.update(extra)
        return out

    @staticmethod
    def write_jsonl(path: Path, snapshot: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'a') as f:
                f.write(json.dumps(snapshot) + '\n')
        except Exception:
            pass

    def close(self) -> None:
        if self.handle is not None:
            _try(lambda: pynvml.nvmlShutdown())
            self.handle = None


# ============================================================
# Process helpers
# ============================================================

def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _container_to_host_pid(pid: int) -> Optional[int]:
    """通过 /proc/<pid>/status 的 NSpid 字段把容器 PID 映射到 host PID。

    Linux 容器（kubernetes pod 等）下，/proc/<pid>/status 的 NSpid 行类似
        NSpid:  <host_pid>  <ns1_pid>  <ns2_pid> ...
    若我们已经在容器视角，第一个数字就是宿主机视角的 PID。
    """
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('NSpid:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        host = int(parts[1])
                        return host if host != pid else None
        return None
    except Exception:
        return None


# ============================================================
# Daemon
# ============================================================

def daemon_loop(out_path: Path, device_index: int, interval: float,
                target_pid: Optional[int], stop_when_target_dies: bool,
                max_seconds: Optional[float] = None):
    monitor = GpuMonitor(device_index=device_index, target_pid=target_pid)
    print(f'[gpu_monitor] daemon started: out={out_path} device={device_index} '
          f'interval={interval}s target_pid={target_pid} '
          f'stop_when_target_dies={stop_when_target_dies}', flush=True)
    t0 = time.time()
    n_dead_checks = 0
    try:
        while True:
            snap = monitor.snapshot(phase='monitor')
            GpuMonitor.write_jsonl(out_path, snap)
            # 退出条件
            if max_seconds is not None and (time.time() - t0) > max_seconds:
                print('[gpu_monitor] max_seconds reached, exit', flush=True)
                break
            if stop_when_target_dies and target_pid is not None:
                if not pid_alive(target_pid):
                    n_dead_checks += 1
                    if n_dead_checks >= 2:  # 连续 2 次确认已死再退出，防止 PID 还没建立时误判
                        # 写一条 final
                        snap = monitor.snapshot(phase='monitor_final', extra={
                            'reason': 'target_pid_dead'
                        })
                        GpuMonitor.write_jsonl(out_path, snap)
                        print(f'[gpu_monitor] target pid {target_pid} dead, exit', flush=True)
                        break
                else:
                    n_dead_checks = 0
            time.sleep(interval)
    except KeyboardInterrupt:
        print('[gpu_monitor] interrupted', flush=True)
    finally:
        monitor.close()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')

    pd = sub.add_parser('daemon', help='启动后台采样 daemon')
    pd.add_argument('--out', required=True, help='JSONL 输出路径')
    pd.add_argument('--device', type=int, default=0, help='GPU index')
    pd.add_argument('--interval', type=float, default=5.0, help='采样间隔（秒）')
    pd.add_argument('--pid', type=int, default=None, help='训练进程 PID（用于聚焦其显存）')
    pd.add_argument('--stop-on-pid-death', action='store_true',
                    help='训练进程退出后 daemon 自动退出')
    pd.add_argument('--max-seconds', type=float, default=None,
                    help='最长运行秒数；不设则常驻')

    pone = sub.add_parser('once', help='打印一次快照（不写文件）')
    pone.add_argument('--device', type=int, default=0)
    pone.add_argument('--pid', type=int, default=None)

    args = p.parse_args()

    if args.cmd == 'daemon':
        daemon_loop(
            out_path=Path(args.out),
            device_index=args.device,
            interval=args.interval,
            target_pid=args.pid,
            stop_when_target_dies=args.stop_on_pid_death,
            max_seconds=args.max_seconds,
        )
    elif args.cmd == 'once':
        m = GpuMonitor(device_index=args.device, target_pid=args.pid)
        snap = m.snapshot(phase='once')
        print(json.dumps(snap, indent=2, ensure_ascii=False))
    else:
        p.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

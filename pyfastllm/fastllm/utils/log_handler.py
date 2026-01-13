# -*- coding: utf-8 -*-
"""
FastLLM Python 日志回调模块

提供便捷的日志处理函数，用于显示推理统计信息、进度等。
"""
import sys
import time
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import pyfastllm
except ImportError:
    pyfastllm = None


class LogLevel(Enum):
    """日志级别"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class LogEvent(Enum):
    """日志事件类型 - 与 C++ 端 basellm.h 中定义保持一致"""
    KVCACHE_CONFIG = 0
    KVCACHE_HIT = 1
    KVCACHE_MISS = 2
    PREFILL_PROGRESS = 3
    PREFILL_COMPLETE = 4
    BATCH_STATUS = 5
    GENERAL = 6


@dataclass
class InferenceStats:
    """推理统计信息"""
    input_tokens: int = 0
    output_tokens: int = 0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0
    
    @property
    def prefill_speed(self) -> float:
        """预填充速度 (tokens/s)"""
        return self.input_tokens / self.prefill_time if self.prefill_time > 0 else 0
    
    @property
    def decode_speed(self) -> float:
        """解码速度 (tokens/s)"""
        return self.output_tokens / self.decode_time if self.decode_time > 0 else 0
    
    @property
    def total_speed(self) -> float:
        """总体速度 (tokens/s)"""
        return (self.input_tokens + self.output_tokens) / self.total_time if self.total_time > 0 else 0


# 全局统计信息
_current_stats = InferenceStats()
_inference_start_time: Optional[float] = None
_prefill_start_time: Optional[float] = None


def _default_log_handler(log_data) -> None:
    """
    默认的日志处理函数，显示推理统计信息
    """
    global _current_stats, _inference_start_time, _prefill_start_time
    
    event = log_data.event
    
    if event == pyfastllm.LogEvent.PREFILL_PROGRESS:
        if log_data.total > 0:
            progress = log_data.current / log_data.total * 100
            speed = log_data.speed if log_data.speed > 0 else 0
            print(f"\r[Prefill] {log_data.current}/{log_data.total} ({progress:.1f}%) - {speed:.1f} tokens/s", 
                  end='', flush=True)
            
    elif event == pyfastllm.LogEvent.PREFILL_COMPLETE:
        if _prefill_start_time:
            _current_stats.prefill_time = time.time() - _prefill_start_time
        _current_stats.input_tokens = log_data.total
        print(f"\n[Prefill Complete] {log_data.total} tokens in {_current_stats.prefill_time:.2f}s "
              f"({_current_stats.prefill_speed:.1f} tokens/s)")
        
    elif event == pyfastllm.LogEvent.BATCH_STATUS:
        print(f"[Batch] Active: {log_data.active}, Pending: {log_data.pending}")
        
    elif event == pyfastllm.LogEvent.GENERAL:
        level_str = {
            pyfastllm.LogLevel.DEBUG: "DEBUG",
            pyfastllm.LogLevel.INFO: "INFO",
            pyfastllm.LogLevel.WARNING: "WARN",
            pyfastllm.LogLevel.ERROR: "ERROR"
        }.get(log_data.level, "INFO")
        print(f"[{level_str}] {log_data.tag}: {log_data.message}")
        
    elif event == pyfastllm.LogEvent.KVCACHE_CONFIG:
        print(f"[KVCache] {log_data.message}")
        
    elif event == pyfastllm.LogEvent.KVCACHE_HIT:
        print(f"[KVCache Hit] {log_data.message}")
        
    elif event == pyfastllm.LogEvent.KVCACHE_MISS:
        print(f"[KVCache Miss] {log_data.message}")


def _simple_log_handler(log_data) -> None:
    """
    简单的日志处理函数，只显示关键信息
    """
    global _current_stats, _prefill_start_time
    
    event = log_data.event
    
    if event == pyfastllm.LogEvent.PREFILL_PROGRESS:
        _prefill_start_time = _prefill_start_time or time.time()
        
    elif event == pyfastllm.LogEvent.PREFILL_COMPLETE:
        if _prefill_start_time:
            elapsed = time.time() - _prefill_start_time
            speed = log_data.total / elapsed if elapsed > 0 else 0
            print(f"[Prefill] {log_data.total} tokens ({speed:.1f} t/s)")
            _prefill_start_time = None


def enable_logging(verbose: bool = True) -> None:
    """
    启用日志回调
    
    Args:
        verbose: True 使用详细日志，False 使用简单日志
    """
    if pyfastllm is None:
        print("Warning: pyfastllm not available, logging disabled")
        return
    
    handler = _default_log_handler if verbose else _simple_log_handler
    pyfastllm.set_log_callback(handler)


def disable_logging() -> None:
    """禁用日志回调"""
    if pyfastllm is not None:
        pyfastllm.set_log_callback(None)


def set_custom_handler(handler: Callable) -> None:
    """
    设置自定义日志处理函数
    
    Args:
        handler: 接收 LogData 对象的回调函数
    """
    if pyfastllm is not None:
        pyfastllm.set_log_callback(handler)


def get_current_stats() -> InferenceStats:
    """获取当前推理统计信息"""
    return _current_stats


# 为了兼容性，提供模块级函数
def print_stats() -> None:
    """打印当前统计信息"""
    stats = get_current_stats()
    print(f"Input tokens:  {stats.input_tokens}")
    print(f"Output tokens: {stats.output_tokens}")
    print(f"Prefill time:  {stats.prefill_time:.2f}s ({stats.prefill_speed:.1f} t/s)")
    print(f"Decode time:   {stats.decode_time:.2f}s ({stats.decode_speed:.1f} t/s)")
    print(f"Total time:    {stats.total_time:.2f}s")

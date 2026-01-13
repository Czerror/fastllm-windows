//
// log_handler.h - 日志回调处理模块
// 提供预定义的日志处理器，用于美化 basellm 的日志输出
// 使用前必须先包含 models/basellm.h
//

#pragma once

#ifndef FASTLLM_LOG_HANDLER_H
#define FASTLLM_LOG_HANDLER_H

#include "models/basellm.h"
#include "utils/console.h"
#include <sstream>
#include <iomanip>

namespace fastllm {
namespace log_handler {

/**
 * 默认的美化日志处理器
 * 使用 console 模块的函数美化输出 basellm 的日志
 */
inline void DefaultLogHandler(const LogData& log) {
    switch (log.event) {
        case LogEvent::KVCacheConfig:
            console::printConfig("KV缓存配置", log.message);
            break;
            
        case LogEvent::KVCacheHit:
            console::logInfo(log.tag, log.message);
            break;
            
        case LogEvent::KVCacheMiss:
            console::logWarn(log.tag, log.message);
            break;

        case LogEvent::PrefillProgress: {
            if (log.data.total > 0) {
                // 使用 updateProgressInline 显示进度
                console::updateProgressInline(
                    (double)log.data.current / log.data.total, 30, "Prefill");
                std::cout << " " << log.data.current << "/" << log.data.total 
                          << " (" << std::fixed << std::setprecision(1) << log.data.speed << " tok/s)" << std::flush;
            }
            break;
        }
        
        case LogEvent::PrefillComplete: {
            std::cout << std::endl; // 进度条结束后换行
            std::ostringstream oss;
            oss << "预填充完成: " << log.data.total << " tokens, 耗时 " 
                << std::fixed << std::setprecision(2) << log.data.elapsed << "s, "
                << (int)log.data.speed << " tokens/s";
            console::printInfo(oss.str());
            break;
        }
        
        case LogEvent::BatchStatus: {
            // 追踪上次活跃数，用于检测生成结束
            static int lastActive = 0;
            
            // 生成结束：active 从 >0 变成 0
            if (lastActive > 0 && log.data.active == 0 && log.data.pending == 0) {
                // 清除状态行
                std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
            }
            lastActive = log.data.active;
            
            if (log.data.active > 0 || log.data.pending > 0) {
                std::ostringstream oss;
                oss << "\r  生成中: 活跃 " << log.data.active 
                    << ", 等待 " << log.data.pending;
                if (log.data.contextLen > 0) {
                    oss << ", 上下文 " << log.data.contextLen;
                }
                if (log.data.speed > 0) {
                    oss << ", " << std::fixed << std::setprecision(1) << log.data.speed << " tok/s";
                }
                // 填充空格覆盖旧内容
                std::string line = oss.str();
                if (line.length() < 70) {
                    line += std::string(70 - line.length(), ' ');
                }
                std::cout << line << std::flush;
            }
            break;
        }
        
        default:
            if (!log.message.empty()) {
                if (log.level == LogLevel::Debug) {
                    console::logDebug(log.tag, log.message);
                } else if (log.level == LogLevel::Warn) {
                    console::logWarn(log.tag, log.message);
                } else if (log.level == LogLevel::Error) {
                    console::logError(log.tag, log.message);
                } else {
                    console::logInfo(log.tag, log.message);
                }
            }
            break;
    }
}

/**
 * 简洁日志处理器 - 只输出关键信息，不显示进度条
 */
inline void SimpleLogHandler(const LogData& log) {
    switch (log.event) {
        case LogEvent::PrefillProgress:
            // 简洁模式下不显示进度
            break;
        case LogEvent::PrefillComplete: {
            std::ostringstream oss;
            oss << "预填充: " << log.data.total << " tokens, " << (int)log.data.speed << " tokens/s";
            console::printInfo(oss.str());
            break;
        }
        default:
            if (!log.message.empty()) {
                console::printInfo(log.message);
            }
            break;
    }
}

/**
 * 静默日志处理器 - 不输出任何日志
 */
inline void SilentLogHandler(const LogData& /*log*/) {
    // 什么都不做
}

/**
 * 注册默认美化日志处理器
 * 客户端只需调用此函数即可启用美化日志
 */
inline void EnablePrettyLogging() {
    SetLogCallback(DefaultLogHandler);
}

/**
 * 注册简洁日志处理器
 */
inline void EnableSimpleLogging() {
    SetLogCallback(SimpleLogHandler);
}

/**
 * 禁用日志输出
 */
inline void DisableLogging() {
    SetLogCallback(SilentLogHandler);
}

} // namespace log_handler
} // namespace fastllm

#endif // FASTLLM_LOG_HANDLER_H

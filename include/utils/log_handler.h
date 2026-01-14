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
        case LogEvent::KVCacheConfig: {
            std::ostringstream oss;
            oss << "KV缓存: " << std::fixed << std::setprecision(2) << log.data.kvCacheMB << " MB, "
                << "Token上限: " << log.data.tokenLimit << ", "
                << "提示词上限: " << log.data.promptLimit << ", "
                << "批量上限: " << log.data.maxBatch;
            console::printConfig("KV缓存配置", oss.str());
            break;
        }
            
        case LogEvent::KVCacheHit: {
            std::ostringstream oss;
            oss << "命中前缀缓存: " << log.data.current << " tokens (输入 " << log.data.total 
                << " tokens, 跳过 " << std::fixed << std::setprecision(1) << log.data.skipPercent 
                << "%, 位置: " << log.data.device << ")";
            console::logInfo(log.tag, oss.str());
            break;
        }
            
        case LogEvent::KVCacheMiss: {
            std::ostringstream oss;
            oss << "未命中缓存, 需预填充 " << log.data.total << " tokens (缓存条目数: " << log.data.cacheEntries << ")";
            console::logWarn(log.tag, oss.str());
            break;
        }

        case LogEvent::ModelLoadProgress: {
            if (log.data.total > 0) {
                // 节流：只在百分比变化时更新显示
                static int lastPercent = -1;
                int percent = (log.data.current * 100) / log.data.total;
                if (percent != lastPercent) {
                    lastPercent = percent;
                    double progress = (double)log.data.current / log.data.total;
                    int width = 30;
                    int filled = static_cast<int>(progress * width);
                    
                    // 隐藏光标，整行打印
                    if (console::getAnsiEnabled()) {
                        std::cout << console::CURSOR_HIDE << console::CLEAR_LINE << "\r"
                                  << console::DIM << "加载权重 " << console::RESET << "[";
                        std::cout << console::GREEN;
                        for (int i = 0; i < filled; ++i) std::cout << "#";
                        std::cout << console::RESET << console::DIM;
                        for (int i = filled; i < width; ++i) std::cout << "-";
                        std::cout << console::RESET << "] " << percent << "%" << std::flush;
                    } else {
                        std::cout << "\r加载权重 [";
                        for (int i = 0; i < filled; ++i) std::cout << "#";
                        for (int i = filled; i < width; ++i) std::cout << "-";
                        std::cout << "] " << percent << "%" << std::flush;
                    }
                }
                // 加载完成时重置状态
                if (log.data.current >= log.data.total) {
                    lastPercent = -1;
                }
            }
            break;
        }
        
        case LogEvent::ModelLoadComplete: {
            // 清除进度行，显示光标，换行准备显示 Warmup
            if (console::getAnsiEnabled()) {
                std::cout << console::CURSOR_SHOW << console::CLEAR_LINE << "\r" << std::flush;
            } else {
                std::cout << "\r" << std::string(60, ' ') << "\r" << std::flush;
            }
            break;
        }
        
        case LogEvent::WarmUp: {
            // 显示 Warmup 状态（覆盖当前行）
            if (console::getAnsiEnabled()) {
                std::cout << console::CLEAR_LINE << "\r" 
                          << console::DIM << "  预热模型..." << console::RESET << std::flush;
            } else {
                std::cout << "\r预热模型..." << std::flush;
            }
            break;
        }

        case LogEvent::PrefillProgress: {
            if (log.data.total > 0) {
                // 隐藏光标避免闪烁
                if (console::getAnsiEnabled()) {
                    std::cout << console::CURSOR_HIDE << console::CLEAR_LINE << "\r"
                              << console::CYAN << "预填充中 " << console::RESET;
                } else {
                    std::cout << "\r预填充中 ";
                }
                // 显示进度条
                double progress = (double)log.data.current / log.data.total;
                int width = 30;
                int filled = static_cast<int>(progress * width);
                std::cout << "[";
                console::ansi(std::cout, console::GREEN);
                for (int i = 0; i < filled; ++i) std::cout << "#";
                console::reset(std::cout);
                console::ansi(std::cout, console::DIM);
                for (int i = filled; i < width; ++i) std::cout << "-";
                console::reset(std::cout);
                std::cout << "] " << static_cast<int>(progress * 100) << "% "
                          << log.data.current << "/" << log.data.total 
                          << " (" << std::fixed << std::setprecision(1) << log.data.speed << " tok/s)" << std::flush;
            }
            break;
        }
        
        case LogEvent::PrefillComplete: {
            // 显示光标，进度条结束后换行
            if (console::getAnsiEnabled()) {
                std::cout << console::CURSOR_SHOW;
            }
            std::cout << std::endl;
            std::ostringstream oss;
            oss << "预填充完成: " << log.data.total << " tokens, 耗时 " 
                << std::fixed << std::setprecision(2) << log.data.elapsed << "s, "
                << (int)log.data.speed << " tokens/s";
            console::printInfo(oss.str());
            break;
        }
        
        case LogEvent::BatchStatus: {
            // 使用核心模块传递的 isComplete 标志判断生成结束
            if (log.data.isComplete) {
                // 显示生成完成信息
                std::ostringstream oss;
                if (console::getAnsiEnabled()) {
                    oss << console::CLEAR_LINE << "\r" << console::DIM
                        << "  生成完成: 上下文 " << log.data.contextLen;
                    if (log.data.speed > 0) {
                        oss << ", " << std::fixed << std::setprecision(1) << log.data.speed << " tokens/s";
                    }
                    oss << console::RESET << std::endl;
                } else {
                    oss << "\r生成完成: 上下文 " << log.data.contextLen;
                    if (log.data.speed > 0) {
                        oss << ", " << std::fixed << std::setprecision(1) << log.data.speed << " tokens/s";
                    }
                    oss << std::endl;
                }
                std::cout << oss.str() << std::flush;
            } else if (log.data.active > 0 || log.data.pending > 0) {
                std::ostringstream oss;
                if (console::getAnsiEnabled()) {
                    oss << "\r" << console::DIM << "  生成中: 活跃 " << log.data.active 
                        << ", 等待 " << log.data.pending;
                } else {
                    oss << "\r  生成中: 活跃 " << log.data.active 
                        << ", 等待 " << log.data.pending;
                }
                if (log.data.contextLen > 0) {
                    oss << ", 上下文 " << log.data.contextLen;
                }
                if (log.data.speed > 0) {
                    oss << ", " << std::fixed << std::setprecision(1) << log.data.speed << " tokens/s";
                }
                if (console::getAnsiEnabled()) {
                    oss << console::RESET;
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
                // 先清除当前行（可能有进度条）
                if (console::getAnsiEnabled()) {
                    std::cout << console::CLEAR_LINE << "\r" << std::flush;
                }
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

//
// inference_stats.h - 推理统计信息模块
// 提供统一的推理统计结构和输出函数，供 C++ apiserver 和 Python 端共用
//

#ifndef FASTLLM_INFERENCE_STATS_H
#define FASTLLM_INFERENCE_STATS_H

#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>

#include "utils/console.h"

namespace fastllm {

// 推理统计信息结构
struct InferenceStatsInfo {
    int promptTokens = 0;          // 提示词 token 数
    int outputTokens = 0;          // 输出 token 数
    double totalTime = 0.0;        // 总耗时（秒）
    double firstTokenTime = 0.0;   // 首字延迟（秒）
    double speed = 0.0;            // 生成速度（tokens/s）
};

// 扩展 console 命名空间，添加推理统计专用输出函数
namespace console {

// 显示推理统计信息（线程安全，防止颜色串行）
inline void printStats(int promptTokens, int outputTokens, double totalTime, double firstTokenTime, double speed) {
    // 注意：调用方（InferenceStatsHelper::print）已清除行并输出"请求完成"
    std::ostringstream oss;
    if (getAnsiEnabled()) {
        oss << GREEN << ICON_CHECK << RESET 
            << " 提示词: " << BRIGHT_CYAN << promptTokens << RESET
            << ", 输出: " << BRIGHT_CYAN << outputTokens << RESET
            << ", 耗时: " << YELLOW << std::fixed << std::setprecision(2) << totalTime << "s" << RESET
            << ", 首字: " << YELLOW << std::fixed << std::setprecision(2) << firstTokenTime << "s" << RESET
            << ", 速度: " << BRIGHT_GREEN << std::fixed << std::setprecision(1) << speed << " tokens/s" << RESET
            << std::endl;
    } else {
        oss << "[完成] 提示词: " << promptTokens 
            << ", 输出: " << outputTokens 
            << ", 耗时: " << std::fixed << std::setprecision(2) << totalTime << "s"
            << ", 首字: " << std::fixed << std::setprecision(2) << firstTokenTime << "s"
            << ", 速度: " << std::fixed << std::setprecision(1) << speed << " tokens/s"
            << std::endl;
    }
    // 一次性输出，避免多线程交错导致颜色串行
    std::cout << oss.str() << std::flush;
}

// 使用 InferenceStatsInfo 结构打印统计信息
inline void printStats(const InferenceStatsInfo& stats) {
    printStats(stats.promptTokens, stats.outputTokens, stats.totalTime, stats.firstTokenTime, stats.speed);
}

} // namespace console

// 推理统计助手类 - 用于统一各端点的性能统计
class InferenceStatsHelper {
public:
    std::chrono::high_resolution_clock::time_point requestStart;
    std::chrono::high_resolution_clock::time_point firstTokenTime;
    int promptTokens = 0;
    int outputTokens = 0;
    bool firstTokenReceived = false;
    long long clientId = 0;  // 请求 ID，用于显示"请求 X 处理完成"
    
    InferenceStatsHelper(int promptToks = 0, long long client = 0) : promptTokens(promptToks), clientId(client) {
        requestStart = std::chrono::high_resolution_clock::now();
    }
    
    // 记录收到第一个 token
    void onFirstToken() {
        if (!firstTokenReceived) {
            firstTokenReceived = true;
            firstTokenTime = std::chrono::high_resolution_clock::now();
        }
    }
    
    // 记录输出 token
    void onToken() {
        onFirstToken();
        outputTokens++;
    }
    
    // 获取首字延迟（秒）
    double getFirstTokenLatency() const {
        if (!firstTokenReceived) return 0;
        return std::chrono::duration<double>(firstTokenTime - requestStart).count();
    }
    
    // 获取总耗时（秒）
    double getTotalTime() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - requestStart).count();
    }
    
    // 获取生成速度（tokens/s，排除首字延迟）
    double getSpeed() const {
        double generateTime = getTotalTime() - getFirstTokenLatency();
        return (outputTokens > 0 && generateTime > 0) ? outputTokens / generateTime : 0;
    }
    
    // 转换为 InferenceStatsInfo
    InferenceStatsInfo toStatsInfo() const {
        InferenceStatsInfo info;
        info.promptTokens = promptTokens;
        info.outputTokens = outputTokens;
        info.totalTime = getTotalTime();
        info.firstTokenTime = getFirstTokenLatency();
        info.speed = getSpeed();
        return info;
    }
    
    // 打印统计信息（先输出"请求完成"，再输出统计）
    void print() const {
        // 先输出"请求完成"（灰色，有缩进）
        if (clientId > 0) {
            if (console::getAnsiEnabled()) {
                printf("\033[2K\r\033[2m  请求 %lld 处理完成\033[0m\n", clientId);
            } else {
                printf("\r%70s\r  请求 %lld 处理完成\n", "", clientId);
            }
        }
        console::printStats(promptTokens, outputTokens, getTotalTime(), getFirstTokenLatency(), getSpeed());
    }
};

} // namespace fastllm

#endif // FASTLLM_INFERENCE_STATS_H

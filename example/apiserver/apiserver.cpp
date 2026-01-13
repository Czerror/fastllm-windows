// Provide by Jacques CHEN (http://whchen.net/index.php/About.html)
// HTML file reference from ChatGLM-MNN （https://github.com/wangzhaode/ChatGLM-MNN)

#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <thread>
#include <stdlib.h>
#include <string>
#include <mutex>

#include <chrono>
#include <optional>

/*
 * Headers
 */

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif //_CRT_SECURE_NO_WARNINGS

#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE
#endif //_CRT_NONSTDC_NO_DEPRECATE

#if defined(_MSC_VER)
#if _MSC_VER < 1900
#error Sorry, Visual Studio versions prior to 2015 are not supported
#endif

#pragma comment(lib, "ws2_32.lib")

#ifdef _WIN64
using ssize_t = __int64;
#else
using ssize_t = long;
#endif
#endif // _MSC_VER

#ifndef S_ISREG
#define S_ISREG(m) (((m)&S_IFREG) == S_IFREG)
#endif // S_ISREG

#ifndef S_ISDIR
#define S_ISDIR(m) (((m)&S_IFDIR) == S_IFDIR)
#endif // S_ISDIR

#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <io.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#ifndef WSA_FLAG_NO_HANDLE_INHERIT
#define WSA_FLAG_NO_HANDLE_INHERIT 0x80
#endif

#ifndef strcasecmp
#define strcasecmp _stricmp
#endif // strcasecmp

using socket_t = SOCKET;
#ifdef CPPHTTPLIB_USE_POLL
#define poll(fds, nfds, timeout) WSAPoll(fds, nfds, timeout)
#endif

#else // not _WIN32

#include <arpa/inet.h>
#ifndef _AIX
#include <ifaddrs.h>
#endif
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#ifdef __linux__
#include <resolv.h>
#endif
#include <netinet/tcp.h>
#ifdef CPPHTTPLIB_USE_POLL
#include <poll.h>
#endif
#include <csignal>
#include <pthread.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using socket_t = int;
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#endif //_WIN32

// 跨平台关闭 socket 的宏
#ifdef _WIN32
#define CloseSocket(s) closesocket(s)
#else
#define CloseSocket(s) close(s)
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include "model.h"

// ===== 控制台美化输出支持 =====
#include "utils/inference_stats.h"
#include "utils/console.h"
#include "utils/log_handler.h"  // 日志回调处理模块

// 本地别名，保持兼容性
using InferenceStats = fastllm::InferenceStatsHelper;
namespace console = fastllm::console;
namespace log_handler = fastllm::log_handler;

long long _GetCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

std::string GenerateRandomID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    for (int i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            ss << '-';
        }
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

static const size_t CHUNK_HEADER_MAX_BYTES = 64;
static const char *HTTP_CRLF = "\r\n";

static bool WriteAll(socket_t fd, const char *data, size_t len);
static bool WriteAll(socket_t fd, const std::string &s);

struct ValidationResult {
    bool ok = true;
    std::string message;
    std::string param;
};

static ValidationResult ValidateTemperature(double v) {
    if (v < 0.0 || v > 2.0) {
        return {false, "temperature must be between 0 and 2", "temperature"};
    }
    return {};
}

static ValidationResult ValidateTopP(double v) {
    if (v < 0.0 || v > 1.0) {
        return {false, "top_p must be between 0 and 1", "top_p"};
    }
    return {};
}

static ValidationResult ValidatePenalty(double v, const std::string &paramName) {
    if (v < -2.0 || v > 2.0) {
        return {false, paramName + " must be between -2 and 2", paramName};
    }
    return {};
}

static json11::Json BuildOpenAIError(const std::string &message,
                                    const std::string &type,
                                    const json11::Json &param = nullptr,
                                    const json11::Json &code = nullptr) {
    return json11::Json::object {
        {"error", json11::Json::object {
            {"message", message},
            {"type", type},
            {"param", param},
            {"code", code}
        }}
    };
}

static std::string HttpStatusText(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 500: return "Internal Server Error";
        default: return "OK";
    }
}

static void SendJson(socket_t client, int status, const json11::Json &body) {
    std::string message;
    message += "HTTP/1.1 " + std::to_string(status) + " " + HttpStatusText(status) + "\r\n";
    message += "Content-Type:application/json\r\n";
    message += "server:fastllm api server\r\n";
    message += "Access-Control-Allow-Origin: *\r\n";
    message += "\r\n";
    message += body.dump();
    (void)WriteAll(client, message);
}

static void SendSSEHeaders(socket_t client) {
    std::string message;
    message += "HTTP/1.1 200 OK\r\n";
    message += "Content-Type:text/event-stream\r\n";
    message += "Cache-Control:no-cache\r\n";
    message += "Connection:keep-alive\r\n";
    message += "server:fastllm api server\r\n";
    message += "Access-Control-Allow-Origin: *\r\n";
    message += "Transfer-Encoding: chunked\r\n";
    message += "\r\n";
    (void)WriteAll(client, message);
}

static bool SendChunk(socket_t client, const std::string &payload) {
    char chunk_header[CHUNK_HEADER_MAX_BYTES];
    sprintf(chunk_header, "%zx\r\n", payload.size());
    return WriteAll(client, chunk_header, strlen(chunk_header))
        && WriteAll(client, payload)
        && WriteAll(client, HTTP_CRLF, 2);
}

// Convert multi-line JSON to single-line JSON for SSE compatibility
// SSE requires each data line to start with "data: " or the entire message on one line
static std::string CompactJson(const std::string &json) {
    std::string result;
    result.reserve(json.size());
    bool inString = false;
    bool escape = false;
    for (char c : json) {
        if (escape) {
            result += c;
            escape = false;
            continue;
        }
        if (c == '\\' && inString) {
            result += c;
            escape = true;
            continue;
        }
        if (c == '"') {
            inString = !inString;
            result += c;
            continue;
        }
        if (!inString) {
            // Skip whitespace outside strings (including newlines and tabs)
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                continue;
            }
        }
        result += c;
    }
    return result;
}

// Check if a string ends with incomplete UTF-8 sequence and return the number of incomplete bytes
// Returns 0 if the string ends with a complete UTF-8 character
static size_t GetIncompleteUtf8Bytes(const std::string &s) {
    if (s.empty()) return 0;
    
    // Check the last few bytes for incomplete UTF-8 sequences
    size_t len = s.size();
    
    // Check for incomplete sequences by looking at the last bytes
    // UTF-8 encoding:
    // 1 byte:  0xxxxxxx (ASCII, always complete)
    // 2 bytes: 110xxxxx 10xxxxxx
    // 3 bytes: 1110xxxx 10xxxxxx 10xxxxxx
    // 4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    
    // Look backwards for the start of a multi-byte sequence
    for (size_t i = 1; i <= 4 && i <= len; i++) {
        unsigned char c = (unsigned char)s[len - i];
        
        // If this is a continuation byte (10xxxxxx), keep looking
        if ((c & 0xC0) == 0x80) {
            continue;
        }
        
        // Found the start byte
        int expectedLen = 0;
        if ((c & 0x80) == 0x00) expectedLen = 1;       // ASCII
        else if ((c & 0xE0) == 0xC0) expectedLen = 2;  // 110xxxxx
        else if ((c & 0xF0) == 0xE0) expectedLen = 3;  // 1110xxxx
        else if ((c & 0xF8) == 0xF0) expectedLen = 4;  // 11110xxx
        else return 0;  // Invalid UTF-8 start byte
        
        // Check if we have all the bytes
        if (i < (size_t)expectedLen) {
            return i;  // Incomplete: we have i bytes but need expectedLen
        }
        return 0;  // Complete
    }
    
    return 0;  // Should not reach here for valid UTF-8
}

// Split string into complete UTF-8 portion and incomplete trailing bytes
static std::pair<std::string, std::string> SplitUtf8(const std::string &s) {
    size_t incomplete = GetIncompleteUtf8Bytes(s);
    if (incomplete == 0) {
        return {s, ""};
    }
    return {s.substr(0, s.size() - incomplete), s.substr(s.size() - incomplete)};
}

// Validate UTF-8 and return the length of valid UTF-8 prefix
// Reference: llama.cpp's validate_utf8() implementation
static size_t ValidateUtf8(const std::string &text) {
    size_t len = 0;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = (unsigned char)text[i];
        size_t charLen = 0;
        
        if ((c & 0x80) == 0x00) {
            // ASCII (0xxxxxxx)
            charLen = 1;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx)
            charLen = 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx)
            charLen = 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx)
            charLen = 4;
        } else {
            // Invalid start byte, truncate here
            break;
        }
        
        // Check if we have enough bytes
        if (i + charLen > text.size()) {
            break;  // Incomplete sequence at end
        }
        
        // Validate continuation bytes (must be 10xxxxxx)
        bool valid = true;
        for (size_t j = 1; j < charLen; j++) {
            if (((unsigned char)text[i + j] & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        
        if (!valid) {
            break;
        }
        
        len = i + charLen;
        i += charLen;
    }
    return len;
}

// Check if a string is valid UTF-8
// Reference: llama.cpp's is_valid_utf8()
static bool IsValidUtf8(const std::string &str) {
    return ValidateUtf8(str) == str.size();
}

// Truncate string to valid UTF-8 boundary
// Reference: llama.cpp's approach to ensure valid UTF-8 in JSON output
static std::string TruncateToValidUtf8(const std::string &s) {
    size_t validLen = ValidateUtf8(s);
    if (validLen == s.size()) {
        return s;
    }
    return s.substr(0, validLen);
}

static bool SendSSEData(socket_t client, const json11::Json &obj) {
    std::string cur = "data: " + CompactJson(obj.dump()) + "\n\n";
    return SendChunk(client, cur);
}

static bool SendSSEDone(socket_t client) {
    std::string cur = "data: [DONE]\n\n";
    return SendChunk(client, cur);
}

static std::string ExtractContentText(const json11::Json &content) {
    if (content.is_string()) {
        return content.string_value();
    }
    if (content.is_null()) {
        return "";
    }
    if (content.is_array()) {
        std::string out;
        for (auto &part : content.array_items()) {
            if (part.is_string()) {
                out += part.string_value();
                continue;
            }
            if (!part.is_object()) {
                continue;
            }
            std::string type = part["type"].string_value();
            if (type == "text" || type == "input_text") {
                if (part["text"].is_string()) {
                    out += part["text"].string_value();
                }
            }
        }
        return out;
    }
    if (content.is_object()) {
        if (content["text"].is_string()) {
            return content["text"].string_value();
        }
    }
    return "";
}

static std::string ExtractMessageText(const json11::Json &msg) {
    if (!msg.is_object()) {
        return "";
    }
    return ExtractContentText(msg["content"]);
}

// ========== Tool Calls 支持 ==========

struct ToolCallInfo {
    std::string id;
    std::string name;
    std::string arguments;
};

// 生成 tool call ID
static std::string GenerateToolCallID() {
    return "call_" + GenerateRandomID().substr(0, 24);
}

// 尝试从模型输出中解析工具调用
// 支持多种格式：
// 1. OpenAI function calling 格式: {"name": "xxx", "arguments": {...}}
// 2. JSON 格式的函数调用数组
static std::vector<ToolCallInfo> ParseToolCalls(const std::string &output) {
    std::vector<ToolCallInfo> calls;
    
    // 查找 JSON 对象的起始位置
    size_t jsonStart = output.find('{');
    if (jsonStart == std::string::npos) {
        return calls;
    }
    
    // 尝试提取最外层的 JSON
    int depth = 0;
    size_t jsonEnd = std::string::npos;
    for (size_t i = jsonStart; i < output.size(); i++) {
        if (output[i] == '{') depth++;
        else if (output[i] == '}') {
            depth--;
            if (depth == 0) {
                jsonEnd = i + 1;
                break;
            }
        }
    }
    
    if (jsonEnd == std::string::npos) {
        return calls;
    }
    
    std::string jsonStr = output.substr(jsonStart, jsonEnd - jsonStart);
    std::string err;
    auto parsed = json11::Json::parse(jsonStr, err);
    
    if (!err.empty() || parsed.is_null()) {
        return calls;
    }
    
    // 检查是否为 function call 格式
    if (parsed["name"].is_string()) {
        ToolCallInfo info;
        info.id = GenerateToolCallID();
        info.name = parsed["name"].string_value();
        if (parsed["arguments"].is_object()) {
            info.arguments = parsed["arguments"].dump();
        } else if (parsed["arguments"].is_string()) {
            info.arguments = parsed["arguments"].string_value();
        } else if (parsed["parameters"].is_object()) {
            info.arguments = parsed["parameters"].dump();
        }
        calls.push_back(info);
    }
    // 检查是否为 tool_calls 数组格式
    else if (parsed["tool_calls"].is_array()) {
        for (auto &item : parsed["tool_calls"].array_items()) {
            if (item["function"].is_object()) {
                ToolCallInfo info;
                info.id = item["id"].is_string() ? item["id"].string_value() : GenerateToolCallID();
                info.name = item["function"]["name"].string_value();
                if (item["function"]["arguments"].is_object()) {
                    info.arguments = item["function"]["arguments"].dump();
                } else if (item["function"]["arguments"].is_string()) {
                    info.arguments = item["function"]["arguments"].string_value();
                }
                calls.push_back(info);
            }
        }
    }
    
    return calls;
}

// 构建 tool_calls JSON 数组
static json11::Json::array BuildToolCallsJson(const std::vector<ToolCallInfo> &calls) {
    json11::Json::array arr;
    for (auto &call : calls) {
        arr.push_back(json11::Json::object {
            {"id", call.id},
            {"type", "function"},
            {"function", json11::Json::object {
                {"name", call.name},
                {"arguments", call.arguments}
            }}
        });
    }
    return arr;
}

// 检查请求中是否包含 tools 定义
static bool HasToolsInRequest(const json11::Json &config) {
    return config["tools"].is_array() && !config["tools"].array_items().empty();
}

// 获取 tool_choice 设置
static std::string GetToolChoice(const json11::Json &config) {
    if (config["tool_choice"].is_string()) {
        return config["tool_choice"].string_value();
    }
    if (config["tool_choice"].is_object()) {
        // 如果指定了具体的 function，返回 "required"
        return "required";
    }
    return "auto"; // 默认值
}

// ========== Response Format 支持 ==========

struct ResponseFormatInfo {
    std::string type;  // "text", "json_object", "json_schema"
    std::string schema; // JSON schema (如果 type 是 json_schema)
};

// 解析 response_format 参数
static ResponseFormatInfo ParseResponseFormat(const json11::Json &config) {
    ResponseFormatInfo info;
    info.type = "text"; // 默认值
    
    if (!config["response_format"].is_object()) {
        return info;
    }
    
    auto rf = config["response_format"];
    if (rf["type"].is_string()) {
        info.type = rf["type"].string_value();
    }
    
    // 解析 json_schema
    if (info.type == "json_schema" && rf["json_schema"].is_object()) {
        auto schema = rf["json_schema"]["schema"];
        if (schema.is_object()) {
            info.schema = schema.dump();
        }
    }
    
    return info;
}

// 构建 JSON 模式提示（添加到 system prompt 中）
static std::string BuildJsonModePrompt(const ResponseFormatInfo &format) {
    if (format.type == "json_object") {
        return "\n\nYou must respond with valid JSON only. Do not include any text outside of the JSON object.";
    }
    if (format.type == "json_schema" && !format.schema.empty()) {
        return "\n\nYou must respond with valid JSON that follows this schema:\n" + format.schema + "\n\nDo not include any text outside of the JSON object.";
    }
    return "";
}

// ========== Tools Prompt 构建 ==========

// 将 tools 定义转换为 prompt 格式，注入到 system message 中
static std::string BuildToolsPrompt(const json11::Json &config) {
    if (!HasToolsInRequest(config)) {
        return "";
    }
    
    std::string toolChoice = GetToolChoice(config);
    
    std::stringstream ss;
    ss << "\n\n# Tools\n\n";
    ss << "You have access to the following tools:\n\n";
    
    for (auto &tool : config["tools"].array_items()) {
        if (tool["type"].string_value() == "function" && tool["function"].is_object()) {
            auto func = tool["function"];
            std::string name = func["name"].string_value();
            std::string desc = func["description"].string_value();
            
            ss << "## " << name << "\n\n";
            if (!desc.empty()) {
                ss << desc << "\n\n";
            }
            
            // 添加参数 schema
            if (func["parameters"].is_object()) {
                ss << "Parameters:\n```json\n" << func["parameters"].dump() << "\n```\n\n";
            }
        }
    }
    
    // 添加工具调用格式说明
    ss << "# Tool Call Format\n\n";
    ss << "When you need to use a tool, respond with a JSON object in this exact format:\n";
    ss << "```json\n";
    ss << "{\n";
    ss << "  \"name\": \"tool_name\",\n";
    ss << "  \"arguments\": { ... }\n";
    ss << "}\n";
    ss << "```\n\n";
    
    if (toolChoice == "required") {
        ss << "You MUST use one of the available tools to respond.\n";
    } else if (toolChoice == "none") {
        ss << "Do NOT use any tools. Respond directly with text.\n";
    } else {
        ss << "Use a tool if it helps answer the user's question. Otherwise, respond directly.\n";
    }
    
    return ss.str();
}

static bool WriteAll(socket_t fd, const char *data, size_t len) {
    if (len == 0) {
        return true;
    }
    const char *p = data;
    size_t left = len;
    while (left > 0) {
#ifdef _WIN32
        int written = send(fd, p, static_cast<int>(left), 0);
        if (written == SOCKET_ERROR) {
            return false;
        }
#else
        int written = write(fd, p, static_cast<unsigned int>(left));
        if (written <= 0) {
            return false;
        }
#endif
        p += written;
        left -= static_cast<size_t>(written);
    }
    return true;
}

static bool WriteAll(socket_t fd, const std::string &s) {
    return WriteAll(fd, s.data(), s.size());
}

std::map <std::string, fastllm::DataType> dataTypeDict = {
    {"float32", fastllm::DataType::FLOAT32},
    {"half", fastllm::DataType::FLOAT16},
    {"float16", fastllm::DataType::FLOAT16},
    {"int8", fastllm::DataType::INT8},
    {"int4", fastllm::DataType::INT4_NOZERO},
    {"int4z", fastllm::DataType::INT4},
    {"int4g", fastllm::DataType::INT4_GROUP}
};

static const char* DEFAULT_API_HOST = "127.0.0.1";
static const char* API_SERVER_VERSION = "1.0.0";

struct APIConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string modelName = "fastllm";

    // 可选：用于 /v1/embeddings 的 embedding 模型路径（BertModel）
    std::string embeddingPath = "";

    std::string host = DEFAULT_API_HOST; // 监听地址

    int threads = 4; // 使用的线程数
    bool lowMemMode = false; // 是否使用低内存模式
    bool cudaEmbedding = false; // 是否使用cudaEmbedding
    int port = 8080; // 端口号
    int tokens = -1; // token容量限制
    int batch = 256; // batch数限制
    int chunkedPrefillSize = -1; // Chunked Prefill 分块大小，-1 表示自动
    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    fastllm::DataType atype = fastllm::DataType::FLOAT32;
    int groupCnt = -1;

    std::map <std::string, int> devices;
    std::map <std::string, int> moeDevices;  // MoE 专家层设备映射

    // API Key 认证 (可选，为空表示不启用认证)
    std::string apiKey = "";

    // 开发模式 (启用 /v1/cancel, /v1/active_conversations 等调试接口)
    bool devMode = false;
};
APIConfig config;

void ToNext(char * &cur, const std::string &target, std::string &v) {
    v = "";
    while (*cur != 0) {
        bool stop = true;
        for (int i = 0; i < target.size(); i++) {
            if (cur[i] != target[i]) {
                stop = false;
                break;
            }
        }
        if (stop && target.size() > 0) {
            cur += target.size();
            break;
        } else {
            v += *(cur++);
        }
    }
}

struct HttpRequest {
    std::string method;
    std::string route;
    std::string type;
    std::unordered_map <std::string, std::string> headers;
    std::string body;

    void Init (char *buffer) {
        char *old = buffer;
        headers.clear();
        ToNext(buffer, " ", method);
        ToNext(buffer, " ", route);
        ToNext(buffer, "\r\n", type);
        while (true) {
            if (buffer[0] == 0 || ((long long)(buffer - old)) > 1024 * 1024) {
                break;
            }
            if (buffer[0] == '\r' && buffer[1] == '\n') {
                buffer += 2;
                ToNext(buffer, "", body);
                break;
            } else {
                std::string key;
                ToNext(buffer, ":", key);
                ToNext(buffer, "\r\n", headers[key]);
            }
        }
    }

    bool IsValid (char *buffer, int size) {
        // 添加调试保护
        if (buffer == nullptr || size <= 0) {
            return false;
        }
        char *old = buffer;
        headers.clear();
        ToNext(buffer, " ", method);
        ToNext(buffer, " ", route);
        ToNext(buffer, "\r\n", type);
        while (true) {
            if (buffer[0] == 0 || ((long long)(buffer - old)) > 1024 * 1024) {
                break;
            }
            if (buffer[0] == '\r' && buffer[1] == '\n') {
                // 遇到空行，检查是否需要 body
                if (headers.find("Content-Length") != headers.end()) {
                    if (size - ((long long)(buffer - old)) - 2 >= atoi(headers["Content-Length"].c_str())) {
                        return true;
                    } else {
                        return false;
                    }
                } else {
                    // 没有 Content-Length (如 GET 请求)，空行后即完成
                    return true;
                }
            } else {
                std::string key;
                ToNext(buffer, ":", key);
                ToNext(buffer, "\r\n", headers[key]);
            }
        }
        return false;
    }

    void Print() {
        for (auto &it : headers) {
            printf("%s: %s\n", it.first.c_str(), it.second.c_str());
        }
        printf("body: %s\n", body.c_str());
    }
} httpChecker;

struct WorkNode {
    socket_t client;
    HttpRequest request;
    json11::Json config;
    std::string error;

    void Init(char *buffer, socket_t client) {
        this->client = client;
        request.Init(buffer);
        config = json11::Json::parse(request.body, this->error);
    }
};

// ========== API Key 认证 ==========

// 从请求头中提取 API Key
static std::string ExtractApiKey(const HttpRequest &req) {
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) {
        return "";
    }
    std::string auth = it->second;
    // 去除开头空格
    while (!auth.empty() && auth[0] == ' ') {
        auth = auth.substr(1);
    }
    // 检查 Bearer 前缀
    const std::string prefix = "Bearer ";
    if (auth.size() > prefix.size() && auth.substr(0, prefix.size()) == prefix) {
        return auth.substr(prefix.size());
    }
    return "";
}

// 验证 API Key，返回 true 表示通过
static bool ValidateApiKey(const HttpRequest &req, const std::string &expectedKey) {
    if (expectedKey.empty()) {
        return true; // 未配置 API Key，跳过验证
    }
    std::string providedKey = ExtractApiKey(req);
    return providedKey == expectedKey;
}

// 发送 401 未授权错误
static void SendUnauthorized(socket_t client) {
    SendJson(client, 401, BuildOpenAIError(
        "Incorrect API key provided. You can find your API key at https://platform.openai.com/account/api-keys.",
        "invalid_request_error",
        nullptr,
        "invalid_api_key"
    ));
}

struct WorkQueue {
    std::unique_ptr<fastllm::basellm> model;
    std::unique_ptr<fastllm::BertModel> embeddingModel;
    int maxActivateQueryNumber = 256;
    int activateQueryNumber = 0;
    int totalQueryNumber = 0;
    std::mutex locker;
    std::condition_variable cv;
    std::queue <WorkNode*> q;
    std::thread *loop;

    void Push(char *buffer, socket_t client) {
        locker.lock();
        q.push(new WorkNode());
        q.back()->Init(buffer, client);
        locker.unlock();

        cv.notify_all();
    }

    void Start() {
        loop = new std::thread ([] (WorkQueue *ts) {
            while (true) {
                std::unique_lock <std::mutex> lock(ts->locker);
                if (ts->activateQueryNumber >= ts->maxActivateQueryNumber) {
                    fastllm::MySleep(0);
                    continue;
                }
                if (ts->q.empty()) {
                    ts->cv.wait(lock);
                }

                while (ts->activateQueryNumber < ts->maxActivateQueryNumber && !ts->q.empty()) {
                    WorkNode *now = ts->q.front();
                    ts->q.pop();
                    ts->activateQueryNumber++;

                    ts->totalQueryNumber++;
                    printf("累计请求数 = %d\n", ts->totalQueryNumber);
//printf("activate = %d, q.size() = %d\n", ts->activateQueryNumber, (int) ts->q.size());

                    std::thread *t = new std::thread([](WorkQueue *ts, WorkNode *now) {
                        ts->Deal(now);
                        printf("客户端 %lld 请求处理完成\n", (long long)now->client);
                        ts->locker.lock();
                        ts->activateQueryNumber--;
                        ts->locker.unlock();
                    }, ts, now);
                }
            }
        }, this);
    }

    void Deal(WorkNode *node) {
        auto *req = &node->request;
        
        // 健康检查端点（不需要认证）
        if ((req->route == "/health" || req->route == "/v1/health") && req->method == "GET") {
            std::string message = "";
            message += "HTTP/1.1 200 OK\r\n";
            message += "Content-Type:application/json\r\n";
            message += "server:fastllm api server\r\n";
            message += "\r\n";
            message += "{\"status\":\"healthy\"}";
            (void)WriteAll(node->client, message);
            CloseSocket(node->client);
            return;
        }

        // 版本信息端点（不需要认证）
        if ((req->route == "/version" || req->route == "/version/") && req->method == "GET") {
            SendJson(node->client, 200, json11::Json::object {
                {"version", API_SERVER_VERSION},
                {"engine", "fastllm"}
            });
            CloseSocket(node->client);
            return;
        }

        // API Key 认证检查（跳过健康检查和版本端点）
        if (!ValidateApiKey(*req, ::config.apiKey)) {
            SendUnauthorized(node->client);
            CloseSocket(node->client);
            return;
        }

        // 开发调试接口（需要 --dev_mode 启用）
        if ((req->route == "/v1/cancel" || req->route == "/v1/cancel/") && req->method == "POST") {
            if (!::config.devMode) {
                SendJson(node->client, 404, BuildOpenAIError(
                    "Endpoint /v1/cancel is only available in dev mode. Start with --dev_mode flag.",
                    "invalid_request_error"
                ));
                CloseSocket(node->client);
                return;
            }

            std::string conversationId = node->config["conversation_id"].string_value();
            if (conversationId.empty()) {
                SendJson(node->client, 400, BuildOpenAIError("conversation_id is required", "invalid_request_error", "conversation_id"));
                CloseSocket(node->client);
                return;
            }

            // 简化实现：实际取消逻辑需要维护活跃对话列表
            // 目前只返回成功响应
            SendJson(node->client, 200, json11::Json::object {
                {"status", "cancelled"},
                {"conversation_id", conversationId},
                {"message", "Cancellation request received (note: full cancellation support requires conversation tracking)"}
            });
            CloseSocket(node->client);
            return;
        }

        if ((req->route == "/v1/active_conversations" || req->route == "/v1/active_conversations/") && req->method == "GET") {
            if (!::config.devMode) {
                SendJson(node->client, 404, BuildOpenAIError(
                    "Endpoint /v1/active_conversations is only available in dev mode. Start with --dev_mode flag.",
                    "invalid_request_error"
                ));
                CloseSocket(node->client);
                return;
            }

            // 返回当前活跃查询数
            SendJson(node->client, 200, json11::Json::object {
                {"active_count", this->activateQueryNumber},
                {"max_count", this->maxActivateQueryNumber},
                {"total_processed", this->totalQueryNumber},
                {"conversations", json11::Json::array {}}  // 简化实现，不跟踪具体对话
            });
            CloseSocket(node->client);
            return;
        }

        // /tokenize 端点 - 文本分词
        if ((req->route == "/tokenize" || req->route == "/tokenize/") && req->method == "POST") {
            std::string content = node->config["content"].string_value();
            if (content.empty()) {
                SendJson(node->client, 400, BuildOpenAIError("content is required", "invalid_request_error", "content"));
                CloseSocket(node->client);
                return;
            }

            bool addSpecial = node->config["add_special"].is_bool() && node->config["add_special"].bool_value();
            bool withPieces = node->config["with_pieces"].is_bool() && node->config["with_pieces"].bool_value();

            // 使用 tokenizer 进行分词
            auto inputs = model->weight.tokenizer.Encode(content);
            
            json11::Json::array tokensArray;
            for (int i = 0; i < inputs.Count(0); i++) {
                int tokenId = static_cast<int>(((float *)inputs.cpuData)[i]);
                
                if (withPieces) {
                    // 解码单个 token 获取文本片段
                    std::vector<float> singleToken = {static_cast<float>(tokenId)};
                    std::string piece = model->weight.tokenizer.Decode(
                        fastllm::Data(fastllm::DataType::FLOAT32, {1}, singleToken)
                    );
                    tokensArray.push_back(json11::Json::object {
                        {"id", tokenId},
                        {"piece", piece}
                    });
                } else {
                    tokensArray.push_back(tokenId);
                }
            }

            SendJson(node->client, 200, json11::Json::object {
                {"tokens", tokensArray}
            });
            CloseSocket(node->client);
            return;
        }

        // /detokenize 端点 - 逆分词
        if ((req->route == "/detokenize" || req->route == "/detokenize/") && req->method == "POST") {
            if (!node->config["tokens"].is_array()) {
                SendJson(node->client, 400, BuildOpenAIError("tokens array is required", "invalid_request_error", "tokens"));
                CloseSocket(node->client);
                return;
            }

            std::vector<float> tokenIds;
            for (auto &t : node->config["tokens"].array_items()) {
                if (t.is_number()) {
                    tokenIds.push_back(static_cast<float>(t.int_value()));
                }
            }

            if (tokenIds.empty()) {
                SendJson(node->client, 200, json11::Json::object {
                    {"content", ""}
                });
                CloseSocket(node->client);
                return;
            }

            // 使用 tokenizer 进行逆分词
            std::string content = model->weight.tokenizer.Decode(
                fastllm::Data(fastllm::DataType::FLOAT32, {static_cast<int>(tokenIds.size())}, tokenIds)
            );

            SendJson(node->client, 200, json11::Json::object {
                {"content", content}
            });
            CloseSocket(node->client);
            return;
        }

        // /slots 端点 - 处理槽位状态
        if ((req->route == "/slots" || req->route == "/slots/") && req->method == "GET") {
            json11::Json::array slotsArray;
            
            // 返回当前服务器状态（简化版本）
            json11::Json::object params;
            params["temperature"] = json11::Json(0.8);
            params["top_k"] = json11::Json(40);
            params["top_p"] = json11::Json(0.95);
            params["n_predict"] = json11::Json(-1);

            json11::Json::object nextToken;
            nextToken["has_next_token"] = json11::Json(this->activateQueryNumber > 0);
            nextToken["n_remain"] = json11::Json(-1);
            nextToken["n_decoded"] = json11::Json(0);

            json11::Json::object slot;
            slot["id"] = json11::Json(0);
            slot["is_processing"] = json11::Json(this->activateQueryNumber > 0);
            slot["n_ctx"] = json11::Json(model->tokensLimit > 0 ? model->tokensLimit : 4096);
            slot["params"] = json11::Json(params);
            slot["next_token"] = json11::Json(nextToken);
            slot["active_requests"] = json11::Json(this->activateQueryNumber);
            slot["max_requests"] = json11::Json(this->maxActivateQueryNumber);

            slotsArray.push_back(json11::Json(slot));

            SendJson(node->client, 200, slotsArray);
            CloseSocket(node->client);
            return;
        }

        // /props 端点 - 服务器属性和配置信息
        if ((req->route == "/props" || req->route == "/props/") && req->method == "GET") {
            // 获取 KV Cache 状态
            int kvCacheEntries = 0;
            {
                std::lock_guard<std::mutex> lock(this->model->pastKVCacheManager.locker);
                kvCacheEntries = static_cast<int>(this->model->pastKVCacheManager.memorys.size());
            }

            json11::Json::object kvCache;
            kvCache["total_entries"] = json11::Json(kvCacheEntries);
            kvCache["max_entries"] = json11::Json(this->model->pastKVCacheManager.maxRecordNum);
            
            json11::Json result = json11::Json::object {
                {"model", ::config.modelName},
                {"model_path", ::config.path},
                {"embedding_model_loaded", this->embeddingModel != nullptr},
                {"server_version", API_SERVER_VERSION},
                {"engine", "fastllm"},
                {"default_generation_settings", json11::Json::object {
                    {"max_tokens", 256},
                    {"temperature", 1.0},
                    {"top_p", 1.0},
                    {"top_k", 1},
                    {"repeat_penalty", 1.0},
                    {"repeat_last_n", 64}
                }},
                {"kv_cache", json11::Json(kvCache)},
                {"supported_endpoints", json11::Json::array {
                    "/v1/chat/completions",
                    "/v1/completions",
                    "/v1/embeddings",
                    "/v1/models",
                    "/v1/rerank",
                    "/health",
                    "/v1/health",
                    "/version",
                    "/props",
                    "/tokenize",
                    "/detokenize",
                    "/slots",
                    "/metrics"
                }},
                {"supported_parameters", json11::Json::array {
                    "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens",
                    "frequency_penalty", "presence_penalty", "repetition_penalty",
                    "repeat_last_n", "stream", "stream_options", "response_format",
                    "tools", "tool_choice", "stop"
                }},
                {"capabilities", json11::Json::object {
                    {"streaming", true},
                    {"tool_calls", true},
                    {"response_format", true},
                    {"embeddings", this->embeddingModel != nullptr},
                    {"rerank", this->embeddingModel != nullptr}
                }}
            };
            
            SendJson(node->client, 200, result);
            CloseSocket(node->client);
            return;
        }

        // /metrics 端点 - Prometheus 监控指标
        if ((req->route == "/metrics" || req->route == "/metrics/") && req->method == "GET") {
            std::stringstream ss;
            
            // Prometheus 格式的指标
            ss << "# HELP fastllm_requests_total Total number of requests processed\n";
            ss << "# TYPE fastllm_requests_total counter\n";
            ss << "fastllm_requests_total " << this->totalQueryNumber << "\n";
            ss << "\n";
            
            ss << "# HELP fastllm_requests_processing Number of requests currently being processed\n";
            ss << "# TYPE fastllm_requests_processing gauge\n";
            ss << "fastllm_requests_processing " << this->activateQueryNumber << "\n";
            ss << "\n";
            
            ss << "# HELP fastllm_requests_max Maximum number of concurrent requests\n";
            ss << "# TYPE fastllm_requests_max gauge\n";
            ss << "fastllm_requests_max " << this->maxActivateQueryNumber << "\n";
            ss << "\n";
            
            ss << "# HELP fastllm_queue_size Number of requests waiting in queue\n";
            ss << "# TYPE fastllm_queue_size gauge\n";
            ss << "fastllm_queue_size " << this->q.size() << "\n";
            ss << "\n";
            
            ss << "# HELP fastllm_model_loaded Whether the model is loaded (1) or not (0)\n";
            ss << "# TYPE fastllm_model_loaded gauge\n";
            ss << "fastllm_model_loaded " << (this->model ? 1 : 0) << "\n";
            ss << "\n";
            
            ss << "# HELP fastllm_embedding_model_loaded Whether the embedding model is loaded (1) or not (0)\n";
            ss << "# TYPE fastllm_embedding_model_loaded gauge\n";
            ss << "fastllm_embedding_model_loaded " << (this->embeddingModel ? 1 : 0) << "\n";

            std::string metrics = ss.str();
            
            std::string message;
            message += "HTTP/1.1 200 OK\r\n";
            message += "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n";
            message += "server: fastllm api server\r\n";
            message += "\r\n";
            message += metrics;
            
            (void)WriteAll(node->client, message);
            CloseSocket(node->client);
            return;
        }
        
        // 模型列表端点
        if ((req->route == "/v1/models" || req->route == "/v1/models/") && req->method == "GET") {
            std::string message = "";
            message += "HTTP/1.1 200 OK\r\n";
            message += "Content-Type:application/json\r\n";
            message += "server:fastllm api server\r\n";
            message += "\r\n";
            
            json11::Json result = json11::Json::object {
                {"object", "list"},
                {"data", json11::Json::array {
                    json11::Json::object {
                        {"id", ::config.modelName},
                        {"object", "model"},
                        {"created", _GetCurrentTime()},
                        {"owned_by", "fastllm"}
                    }
                }}
            };
            message += result.dump();
            (void)WriteAll(node->client, message);
            CloseSocket(node->client);
            return;
        }

        // OpenAI embeddings 端点
        if ((req->route == "/v1/embeddings" || req->route == "/v1/embeddings/") && req->method == "POST") {
            if (!this->embeddingModel) {
                SendJson(node->client, 400, BuildOpenAIError(
                    "Embeddings model not loaded. Start apiserver with --embedding_path.",
                    "invalid_request_error",
                    "model"
                ));
                CloseSocket(node->client);
                return;
            }

            std::string inputText;
            if (node->config["input"].is_string()) {
                inputText = node->config["input"].string_value();
            } else if (node->config["input"].is_array() && !node->config["input"].array_items().empty()) {
                auto first = node->config["input"].array_items()[0];
                inputText = first.is_string() ? first.string_value() : first.dump();
            } else {
                SendJson(node->client, 400, BuildOpenAIError("Input cannot be empty", "invalid_request_error", "input"));
                CloseSocket(node->client);
                return;
            }

            // 计算 token 数（尽量使用 tokenizer）
            int promptTokens = 0;
            {
                auto inputs = this->embeddingModel->weight.tokenizer.Encode(inputText);
                promptTokens = inputs.Count(0);
            }

            auto vec = this->embeddingModel->EmbeddingSentence(inputText, true);
            json11::Json::array emb;
            emb.reserve(vec.size());
            for (auto v : vec) {
                emb.push_back(v);
            }

            std::string respModel = node->config["model"].string_value();
            if (respModel.empty()) {
                respModel = ::config.modelName;
            }

            json11::Json result = json11::Json::object {
                {"object", "list"},
                {"data", json11::Json::array {
                    json11::Json::object {
                        {"object", "embedding"},
                        {"embedding", emb},
                        {"index", 0}
                    }
                }},
                {"model", respModel},
                {"usage", json11::Json::object {
                    {"prompt_tokens", promptTokens},
                    {"total_tokens", promptTokens}
                }}
            };

            SendJson(node->client, 200, result);
            CloseSocket(node->client);
            return;
        }

        // OpenAI /v1/rerank 端点 (使用 embedding 模型计算余弦相似度)
        if ((req->route == "/v1/rerank" || req->route == "/v1/rerank/") && req->method == "POST") {
            if (!this->embeddingModel) {
                SendJson(node->client, 400, BuildOpenAIError(
                    "Embeddings model not loaded. Start apiserver with --embedding_path to enable rerank.",
                    "invalid_request_error",
                    "model"
                ));
                CloseSocket(node->client);
                return;
            }

            // 解析请求
            std::string query = node->config["query"].string_value();
            if (query.empty()) {
                SendJson(node->client, 400, BuildOpenAIError("query is required", "invalid_request_error", "query"));
                CloseSocket(node->client);
                return;
            }

            if (!node->config["documents"].is_array() || node->config["documents"].array_items().empty()) {
                SendJson(node->client, 400, BuildOpenAIError("documents is required and must be non-empty array", "invalid_request_error", "documents"));
                CloseSocket(node->client);
                return;
            }

            std::vector<std::string> documents;
            for (auto &doc : node->config["documents"].array_items()) {
                if (doc.is_string()) {
                    documents.push_back(doc.string_value());
                } else if (doc.is_object() && doc["text"].is_string()) {
                    documents.push_back(doc["text"].string_value());
                }
            }

            int topN = node->config["top_n"].is_number() ? node->config["top_n"].int_value() : static_cast<int>(documents.size());
            topN = std::min(topN, static_cast<int>(documents.size()));

            // 计算 query embedding
            auto queryEmb = this->embeddingModel->EmbeddingSentence(query, true);

            // 计算每个文档的 embedding 和相似度
            std::vector<std::pair<int, double>> scores;
            for (size_t i = 0; i < documents.size(); i++) {
                auto docEmb = this->embeddingModel->EmbeddingSentence(documents[i], true);
                
                // 计算余弦相似度
                double dotProduct = 0.0, normA = 0.0, normB = 0.0;
                size_t minLen = std::min(queryEmb.size(), docEmb.size());
                for (size_t j = 0; j < minLen; j++) {
                    dotProduct += queryEmb[j] * docEmb[j];
                    normA += queryEmb[j] * queryEmb[j];
                    normB += docEmb[j] * docEmb[j];
                }
                double similarity = (normA > 0 && normB > 0) ? dotProduct / (std::sqrt(normA) * std::sqrt(normB)) : 0.0;
                scores.push_back({static_cast<int>(i), similarity});
            }

            // 按相似度降序排序
            std::sort(scores.begin(), scores.end(), [](auto &a, auto &b) {
                return a.second > b.second;
            });

            // 构建结果
            json11::Json::array results;
            for (int i = 0; i < topN && i < static_cast<int>(scores.size()); i++) {
                auto &item = scores[i];
                results.push_back(json11::Json::object {
                    {"index", item.first},
                    {"relevance_score", item.second},
                    {"document", json11::Json::object {
                        {"text", documents[item.first]}
                    }}
                });
            }

            std::string respModel = node->config["model"].string_value();
            if (respModel.empty()) {
                respModel = ::config.modelName;
            }

            json11::Json response = json11::Json::object {
                {"object", "list"},
                {"data", results},
                {"model", respModel},
                {"usage", json11::Json::object {
                    {"total_tokens", 0}  // 简化实现，不计算具体 token 数
                }}
            };

            SendJson(node->client, 200, response);
            CloseSocket(node->client);
            return;
        }
        
        if ((req->route == "/generate" || req->route == "/generate/") && req->method == "POST") {
            std::string message = "";
            message += "HTTP/1.1 200 OK\r\n";
            message += "Content-Type:application/json\r\n";
            message += "server:fastllm api server\r\n";
            message += "\r\n";

            if (node->error == "") {
                if (node->config["prompt"].is_null()) {
                    node->error = "prompt is empty!";
                }
            }
            if (node->error != "") {
                printf("error body = %s, prompt = %s, error = %s\n", node->request.body.c_str(), node->config["prompt"].string_value().c_str(), node->error.c_str());
                message += node->error;
                (void)WriteAll(node->client, message);
                CloseSocket(node->client);
                return;
            }

            std::string output = "";
            fastllm::ChatMessages messages;
            std::string promptText = node->config["prompt"].string_value();
            messages.push_back({"user", promptText});
            auto prompt = model->ApplyChatTemplate(messages);
            auto inputs = model->weight.tokenizer.Encode(prompt);
            std::vector<int> tokens;
            for (int i = 0; i < inputs.Count(0); i++) {
                tokens.push_back(((float *) inputs.cpuData)[i]);
            }
            fastllm::GenerationConfig config;
            config.output_token_limit = node->config["max_tokens"].is_null() ? 200 : node->config["max_tokens"].int_value();
            int handleId = model->LaunchResponseTokens(tokens, config);
            std::vector<float> results;
            while (true) {
                int result = model->FetchResponseTokens(handleId);
                if (result == -1) {
                    break;
                } else {
                    results.clear();
                    results.push_back(result);
                    output += model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));

                    std::string cur = (message + output);
                    if (!WriteAll(node->client, cur)) {
                        model->AbortResponse(handleId);
                        CloseSocket(node->client);
                        return;
                    }
                }
            }

            message += output;
            (void)WriteAll(node->client, message);

            CloseSocket(node->client);
        } else if ((req->route == "/v1/chat/completions" || req->route == "/v1/chat/completions/") && req->method == "POST") {
            std::string message = "";
            message += "HTTP/1.1 200 OK\r\n";
            message += "Content-Type:application/json\r\n";
            message += "server:fastllm api server\r\n";
            message += "\r\n";

            fastllm::ChatMessages chatMessages;
            if (node->config["messages"].is_array()) {
                for (auto &it : node->config["messages"].array_items()) {
                    std::string role = it["role"].string_value();
                    std::string content = ExtractMessageText(it);
                    chatMessages.push_back({role, content});
                }
            } else if (node->config["prompt"].is_string()) {
                std::string promptText = node->config["prompt"].string_value();
                chatMessages.push_back({"user", promptText});
            } else {
                node->error = "messages or prompt is required";
            }

            {
                std::string reqModel = node->config["model"].string_value();
                if (!reqModel.empty() && reqModel != ::config.modelName) {
                    node->error = "The model `" + reqModel + "` does not exist.";
                }
            }

            json11::Json errorParam = nullptr;
            int errorStatus = 400;
            std::string errorType = "invalid_request_error";

            {
                std::string reqModel = node->config["model"].string_value();
                if (!reqModel.empty() && reqModel != ::config.modelName) {
                    errorStatus = 404;
                    errorType = "model_not_found";
                }
            }
            if (node->error == "messages or prompt is required") {
                errorParam = "messages";
            }

            if (node->error == "" && node->config["temperature"].is_number()) {
                auto vr = ValidateTemperature(node->config["temperature"].number_value());
                if (!vr.ok) {
                    node->error = vr.message;
                    errorParam = vr.param;
                }
            }
            if (node->error == "" && node->config["top_p"].is_number()) {
                auto vr = ValidateTopP(node->config["top_p"].number_value());
                if (!vr.ok) {
                    node->error = vr.message;
                    errorParam = vr.param;
                }
            }
            if (node->error == "" && node->config["frequency_penalty"].is_number()) {
                auto vr = ValidatePenalty(node->config["frequency_penalty"].number_value(), "frequency_penalty");
                if (!vr.ok) {
                    node->error = vr.message;
                    errorParam = vr.param;
                }
            }
            if (node->error == "" && node->config["presence_penalty"].is_number()) {
                auto vr = ValidatePenalty(node->config["presence_penalty"].number_value(), "presence_penalty");
                if (!vr.ok) {
                    node->error = vr.message;
                    errorParam = vr.param;
                }
            }

            if (node->error != "") {
                SendJson(node->client, errorStatus, BuildOpenAIError(node->error, errorType, errorParam));
                CloseSocket(node->client);
                return;
            }

            // 处理 tools - 将工具定义注入到 system prompt
            std::string toolsPrompt = BuildToolsPrompt(node->config);
            if (!toolsPrompt.empty()) {
                // 查找是否已有 system 消息
                bool hasSystem = false;
                for (auto &msg : chatMessages) {
                    if (msg.first == "system") {
                        msg.second += toolsPrompt;
                        hasSystem = true;
                        break;
                    }
                }
                if (!hasSystem) {
                    // 在开头添加 system 消息
                    chatMessages.insert(chatMessages.begin(), {"system", toolsPrompt.substr(2)}); // 去掉开头的 \n\n
                }
            }

            // 处理 response_format
            auto responseFormat = ParseResponseFormat(node->config);
            if (responseFormat.type == "json_object" || responseFormat.type == "json_schema") {
                // 向 system prompt 注入 JSON 格式要求
                std::string jsonPrompt = BuildJsonModePrompt(responseFormat);
                if (!jsonPrompt.empty()) {
                    // 查找是否已有 system 消息
                    bool hasSystem = false;
                    for (auto &msg : chatMessages) {
                        if (msg.first == "system") {
                            msg.second += jsonPrompt;
                            hasSystem = true;
                            break;
                        }
                    }
                    if (!hasSystem) {
                        // 在开头添加 system 消息
                        chatMessages.insert(chatMessages.begin(), {"system", jsonPrompt.substr(2)}); // 去掉开头的 \n\n
                    }
                }
            }

            auto prompt = model->ApplyChatTemplate(chatMessages);
            auto inputs = model->weight.tokenizer.Encode(prompt);
            std::vector<int> tokens;
            for (int i = 0; i < inputs.Count(0); i++) {
                tokens.push_back(((float *) inputs.cpuData)[i]);
            }

            fastllm::GenerationConfig config;
            if (node->config["max_tokens"].is_number()) {
                config.output_token_limit = node->config["max_tokens"].int_value();
            } else if (node->config["max_completion_tokens"].is_number()) {
                config.output_token_limit = node->config["max_completion_tokens"].int_value();
            } else {
                config.output_token_limit = 256;
            }
            if (node->config["frequency_penalty"].is_number()) {
                config.repeat_penalty = node->config["frequency_penalty"].number_value();
            }
            if (node->config["temperature"].is_number()) {
                config.temperature = node->config["temperature"].number_value();
            }
            if (node->config["top_p"].is_number()) {
                config.top_p = node->config["top_p"].number_value();
            }
            if (node->config["top_k"].is_number()) {
                config.top_k = node->config["top_k"].int_value();
            }
            // presence_penalty 映射到 repeat_penalty (如果未设置 frequency_penalty)
            if (node->config["presence_penalty"].is_number() && !node->config["frequency_penalty"].is_number()) {
                config.repeat_penalty = 1.0f + node->config["presence_penalty"].number_value();
            }
            // repetition_penalty 直接支持 (兼容 HuggingFace 风格)
            if (node->config["repetition_penalty"].is_number()) {
                config.repeat_penalty = node->config["repetition_penalty"].number_value();
            }
            // last_n 参数 (用于重复惩罚的上下文窗口)
            if (node->config["repeat_last_n"].is_number()) {
                config.last_n = node->config["repeat_last_n"].int_value();
            }

            std::string output = "";
            int handleId = model->LaunchResponseTokens(tokens, config);
            auto requestStartTime = std::chrono::high_resolution_clock::now();  // 记录开始时间
            
            bool isStream = false;
            if (node->config["stream"].is_bool() && node->config["stream"].bool_value()) {
                isStream = true;
            }

            bool includeUsage = true;
            if (node->config["stream_options"].is_object()) {
                if (node->config["stream_options"]["include_usage"].is_bool()) {
                    includeUsage = node->config["stream_options"]["include_usage"].bool_value();
                }
            }

            std::string curId = "fastllm-" + GenerateRandomID();
            auto createTime = _GetCurrentTime();

            if (isStream) {
                message = "";
                message += "HTTP/1.1 200 OK\r\n";
                message += "Content-Type: text/event-stream\r\n";
                message += "Cache-Control: no-cache\r\n";
                message += "Connection: keep-alive\r\n";
                message += "server: fastllm api server\r\n";
                message += "Access-Control-Allow-Origin: *\r\n";
                message += "\r\n";
                
                if (!WriteAll(node->client, message)) {
                    model->AbortResponse(handleId);
                    CloseSocket(node->client);
                    return;
                }
            
                json11::Json startResult = json11::Json::object {
                    {"id", curId},
                    {"object", "chat.completion.chunk"},
                    {"created", createTime},
                    {"model", ::config.modelName},
                    {"system_fingerprint", "fastllm-" + ::config.modelName},
                    {"choices", json11::Json::array {
                        json11::Json::object {
                            {"index", 0},
                            {"delta", json11::Json::object {
                                {"role", "assistant"}
                            }},
                            {"logprobs", nullptr},
                            {"finish_reason", nullptr}
                        }
                    }}
                };
                std::string cur = ("data: " + CompactJson(startResult.dump()) + "\n\n");

                if (!WriteAll(node->client, cur)) {
                    model->AbortResponse(handleId);
                    CloseSocket(node->client);
                    return;
                }

                int outputTokens = 0;
                std::vector<float> results;
                std::string utf8Buffer;  // Buffer for incomplete UTF-8 sequences (reference: llama.cpp)
                InferenceStats stats((int)tokens.size());
                while (true) {
                    int result = model->FetchResponseTokens(handleId);
                    if (result == -1) {
                        // Flush any remaining buffer content before finishing
                        // Reference: llama.cpp flushes incomplete UTF-8 at end
                        if (!utf8Buffer.empty()) {
                            // Send remaining buffer - truncate to valid UTF-8
                            std::string validContent = TruncateToValidUtf8(utf8Buffer);
                            if (!validContent.empty()) {
                                json11::Json bufferResult = json11::Json::object {
                                    {"id", curId},
                                    {"object", "chat.completion.chunk"},
                                    {"created", createTime},
                                    {"model", ::config.modelName},
                                    {"system_fingerprint", "fastllm-" + ::config.modelName},
                                    {"choices", json11::Json::array {
                                        json11::Json::object {
                                            {"index", 0},
                                            {"delta", json11::Json::object {
                                                {"content", validContent}
                                            }},
                                            {"logprobs", nullptr},
                                            {"finish_reason", nullptr}
                                        }
                                    }}
                                };
                                std::string bufCur = ("data: " + CompactJson(bufferResult.dump()) + "\n\n");
                                (void)WriteAll(node->client, bufCur);
                            }
                            utf8Buffer.clear();
                        }
                        
                        std::string finishReason = "stop";
                        if (outputTokens >= config.output_token_limit) {
                            finishReason = "length";
                        }

                        json11::Json partResult = json11::Json::object {
                            {"id", curId},
                            {"object", "chat.completion.chunk"},
                            {"created", createTime},
                            {"model", ::config.modelName},
                            {"system_fingerprint", "fastllm-" + ::config.modelName},
                            {"choices", json11::Json::array {
                                json11::Json::object {
                                    {"index", 0},
                                    {"delta", json11::Json::object {}},
                                    {"logprobs", nullptr},
                                    {"finish_reason", finishReason}
                                }
                            }}
                        };

                        if (includeUsage) {
                            auto obj = partResult.object_items();
                            obj["usage"] = json11::Json::object {
                                {"prompt_tokens", (int)tokens.size()},
                                {"total_tokens", (int)tokens.size() + outputTokens},
                                {"completion_tokens", outputTokens}
                            };
                            partResult = obj;
                        }

                        std::string cur = ("data: " + CompactJson(partResult.dump()) + "\n\n");
                        (void)WriteAll(node->client, cur);
                        break;
                    } else {
                        // 记录首字时间并统计token
                        stats.onToken();
                        outputTokens++;
                        results.clear();
                        results.push_back(result);
                        std::string decoded = model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                        
                        // Skip empty content
                        if (decoded.empty()) {
                            continue;
                        }
                        
                        // UTF-8 buffering: combine with any previous incomplete sequence
                        // Reference: llama.cpp's approach to handle incomplete UTF-8 across token boundaries
                        std::string combined = utf8Buffer + decoded;
                        utf8Buffer.clear();
                        
                        // Split into complete and incomplete UTF-8 portions
                        auto [complete, incomplete] = SplitUtf8(combined);
                        
                        // Store incomplete portion for next iteration
                        utf8Buffer = incomplete;
                        
                        // If no complete content, continue to next token
                        if (complete.empty()) {
                            continue;
                        }
                        
                        // Use the complete UTF-8 portion
                        std::string now = complete;
                        
                        json11::Json partResult = json11::Json::object {
                            {"id", curId},
                            {"object", "chat.completion.chunk"},
                            {"created", createTime},
                            {"model", ::config.modelName},
                            {"system_fingerprint", "fastllm-" + ::config.modelName},
                            {"choices", json11::Json::array {
                                json11::Json::object {
                                    {"index", 0},
                                    {"delta", json11::Json::object {
                                        {"content", now}
                                    }},
                                    {"logprobs", nullptr},
                                    {"finish_reason", nullptr}
                                }
                            }}
                        };

                        std::string cur = ("data: " + CompactJson(partResult.dump()) + "\n\n");
                        
                        if (!WriteAll(node->client, cur)) {
                            model->AbortResponse(handleId);
                            CloseSocket(node->client);
                            return;
                        }
                    }
                }

                cur = ("data: [DONE]\n\n");
                (void)WriteAll(node->client, cur);

                // 显示统计信息
                stats.print();

                CloseSocket(node->client);
            } else {
                InferenceStats stats((int)tokens.size());
                int outputTokens = 0;
                std::vector<float> results;
                while (true) {
                    int result = model->FetchResponseTokens(handleId);
                    if (result == -1) {
                        break;
                    } else {
                        // 记录首字时间并统计token
                        stats.onToken();
                        results.clear();
                        results.push_back(result);
                        output += model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                        outputTokens++;
                    }
                }

                // 判断是否达到 token 限制
                std::string finishReason = "stop";
                if (outputTokens >= config.output_token_limit) {
                    finishReason = "length";
                }

                // 检测 tool_calls
                bool hasTools = HasToolsInRequest(node->config);
                std::vector<ToolCallInfo> toolCalls;
                if (hasTools) {
                    toolCalls = ParseToolCalls(output);
                    if (!toolCalls.empty()) {
                        finishReason = "tool_calls";
                    }
                }

                // 构建消息对象
                json11::Json::object msgObj = {
                    {"role", "assistant"}
                };
                
                if (!toolCalls.empty()) {
                    // 有工具调用时，content 可以为 null 或包含思考过程
                    msgObj["content"] = nullptr;
                    msgObj["tool_calls"] = BuildToolCallsJson(toolCalls);
                } else {
                    msgObj["content"] = output;
                }

                json11::Json result = json11::Json::object {
                    {"id", curId},
                    {"object", "chat.completion"},
                    {"created", createTime},
                    {"model", ::config.modelName},
                    {"system_fingerprint", "fastllm-" + ::config.modelName},
                    {"choices", json11::Json::array {
                        json11::Json::object {
                            {"index", 0},
                            {"message", msgObj},
                            {"logprobs", nullptr},
                            {"finish_reason", finishReason}
                        }
                    }},
                    {"usage", json11::Json::object {
                        {"prompt_tokens", (int)tokens.size()},
                        {"total_tokens", (int)tokens.size() + outputTokens},
                        {"completion_tokens", outputTokens}
                    }}
                };

                message += result.dump();
                (void)WriteAll(node->client, message);

                // 显示统计信息
                stats.print();

                CloseSocket(node->client);
            }
            return;
        } else if ((req->route == "/v1/completions" || req->route == "/v1/completions/") && req->method == "POST") {
            // OpenAI /v1/completions
            std::string promptText;
            if (node->config["prompt"].is_string()) {
                promptText = node->config["prompt"].string_value();
            } else if (node->config["prompt"].is_array() && !node->config["prompt"].array_items().empty()) {
                auto first = node->config["prompt"].array_items()[0];
                promptText = first.is_string() ? first.string_value() : first.dump();
            } else {
                SendJson(node->client, 400, BuildOpenAIError("prompt is required", "invalid_request_error", "prompt"));
                CloseSocket(node->client);
                return;
            }

            {
                std::string reqModel = node->config["model"].string_value();
                if (!reqModel.empty() && reqModel != ::config.modelName) {
                    SendJson(node->client, 404, BuildOpenAIError(
                        "The model `" + reqModel + "` does not exist.",
                        "model_not_found"
                    ));
                    CloseSocket(node->client);
                    return;
                }
            }

            if (node->config["temperature"].is_number()) {
                auto vr = ValidateTemperature(node->config["temperature"].number_value());
                if (!vr.ok) {
                    SendJson(node->client, 400, BuildOpenAIError(vr.message, "invalid_request_error", vr.param));
                    CloseSocket(node->client);
                    return;
                }
            }
            if (node->config["top_p"].is_number()) {
                auto vr = ValidateTopP(node->config["top_p"].number_value());
                if (!vr.ok) {
                    SendJson(node->client, 400, BuildOpenAIError(vr.message, "invalid_request_error", vr.param));
                    CloseSocket(node->client);
                    return;
                }
            }
            if (node->config["frequency_penalty"].is_number()) {
                auto vr = ValidatePenalty(node->config["frequency_penalty"].number_value(), "frequency_penalty");
                if (!vr.ok) {
                    SendJson(node->client, 400, BuildOpenAIError(vr.message, "invalid_request_error", vr.param));
                    CloseSocket(node->client);
                    return;
                }
            }

            fastllm::GenerationConfig gen;
            gen.output_token_limit = node->config["max_tokens"].is_number() ? node->config["max_tokens"].int_value() : 16;
            if (node->config["temperature"].is_number()) {
                gen.temperature = node->config["temperature"].number_value();
            }
            if (node->config["top_p"].is_number()) {
                gen.top_p = node->config["top_p"].number_value();
            }
            if (node->config["top_k"].is_number()) {
                gen.top_k = node->config["top_k"].int_value();
            }
            if (node->config["frequency_penalty"].is_number()) {
                gen.repeat_penalty = node->config["frequency_penalty"].number_value();
            }

            auto inputs = model->weight.tokenizer.Encode(promptText);
            std::vector<int> tokens;
            for (int i = 0; i < inputs.Count(0); i++) {
                tokens.push_back(((float *) inputs.cpuData)[i]);
            }

            const bool echo = node->config["echo"].is_bool() && node->config["echo"].bool_value();
            const bool isStream = node->config["stream"].is_bool() && node->config["stream"].bool_value();

            const std::string curId = "cmpl-" + GenerateRandomID();
            const auto createTime = _GetCurrentTime();

            int handleId = model->LaunchResponseTokens(tokens, gen);

            if (isStream) {
                std::string header;
                header += "HTTP/1.1 200 OK\r\n";
                header += "Content-Type: text/event-stream\r\n";
                header += "Cache-Control: no-cache\r\n";
                header += "Connection: keep-alive\r\n";
                header += "server: fastllm api server\r\n";
                header += "Access-Control-Allow-Origin: *\r\n";
                header += "\r\n";

                if (!WriteAll(node->client, header)) {
                    model->AbortResponse(handleId);
                    CloseSocket(node->client);
                    return;
                }

                if (echo && !promptText.empty()) {
                    json11::Json echoChunk = json11::Json::object {
                        {"id", curId},
                        {"object", "text_completion"},
                        {"created", createTime},
                        {"model", ::config.modelName},
                        {"system_fingerprint", "fastllm-" + ::config.modelName},
                        {"choices", json11::Json::array {
                            json11::Json::object {
                                {"index", 0},
                                {"text", promptText},
                                {"logprobs", nullptr},
                                {"finish_reason", nullptr}
                            }
                        }}
                    };
                    std::string cur = "data: " + CompactJson(echoChunk.dump()) + "\n\n";
                    if (!WriteAll(node->client, cur)) {
                        model->AbortResponse(handleId);
                        CloseSocket(node->client);
                        return;
                    }
                }

                int outputTokens = 0;
                std::vector<float> results;
                InferenceStats stats((int)tokens.size());
                while (true) {
                    int result = model->FetchResponseTokens(handleId);
                    if (result == -1) {
                        std::string finishReason = "stop";
                        if (outputTokens >= gen.output_token_limit) {
                            finishReason = "length";
                        }
                        json11::Json endChunk = json11::Json::object {
                            {"id", curId},
                            {"object", "text_completion"},
                            {"created", createTime},
                            {"model", ::config.modelName},
                            {"system_fingerprint", "fastllm-" + ::config.modelName},
                            {"choices", json11::Json::array {
                                json11::Json::object {
                                    {"index", 0},
                                    {"text", ""},
                                    {"logprobs", nullptr},
                                    {"finish_reason", finishReason}
                                }
                            }}
                        };
                        std::string cur = "data: " + CompactJson(endChunk.dump()) + "\n\n";
                        (void)WriteAll(node->client, cur);
                        break;
                    }
                    stats.onToken();
                    outputTokens++;
                    results.clear();
                    results.push_back(result);
                    std::string now = model->weight.tokenizer.Decode(
                        fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results)
                    );
                    json11::Json part = json11::Json::object {
                        {"id", curId},
                        {"object", "text_completion"},
                        {"created", createTime},
                        {"model", ::config.modelName},
                        {"system_fingerprint", "fastllm-" + ::config.modelName},
                        {"choices", json11::Json::array {
                            json11::Json::object {
                                {"index", 0},
                                {"text", now},
                                {"logprobs", nullptr},
                                {"finish_reason", nullptr}
                            }
                        }}
                    };
                    std::string cur = "data: " + CompactJson(part.dump()) + "\n\n";
                    if (!WriteAll(node->client, cur)) {
                        model->AbortResponse(handleId);
                        CloseSocket(node->client);
                        return;
                    }
                }

                std::string done = "data: [DONE]\n\n";
                (void)WriteAll(node->client, done);
                stats.print();
                CloseSocket(node->client);
                return;
            }

            std::string output;
            if (echo) {
                output = promptText;
            }

            int outputTokens = 0;
            std::vector<float> results;
            InferenceStats stats((int)tokens.size());
            while (true) {
                int result = model->FetchResponseTokens(handleId);
                if (result == -1) {
                    break;
                }
                stats.onToken();
                results.clear();
                results.push_back(result);
                output += model->weight.tokenizer.Decode(
                    fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results)
                );
                outputTokens++;
            }

            std::string finishReason = "stop";
            if (outputTokens >= gen.output_token_limit) {
                finishReason = "length";
            }

            json11::Json result = json11::Json::object {
                {"id", curId},
                {"object", "text_completion"},
                {"created", createTime},
                {"model", ::config.modelName},
                {"system_fingerprint", "fastllm-" + ::config.modelName},
                {"choices", json11::Json::array {
                    json11::Json::object {
                        {"index", 0},
                        {"text", output},
                        {"logprobs", nullptr},
                        {"finish_reason", finishReason}
                    }
                }},
                {"usage", json11::Json::object {
                    {"prompt_tokens", (int)tokens.size()},
                    {"total_tokens", (int)tokens.size() + outputTokens},
                    {"completion_tokens", outputTokens}
                }}
            };

            SendJson(node->client, 200, result);
            stats.print();
            CloseSocket(node->client);
            return;
        } else {
            CloseSocket(node->client);
            return;
        }
    }
} workQueue;

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<--embedding_path> <args>:    embedding模型文件路径(用于 /v1/embeddings，可选)" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<--atype> <args>:             设置推理使用的数据类型(float32/float16)" << std::endl;
    std::cout << "<--batch/--max_batch> <args>: 最大batch数" << std::endl;
    std::cout << "<--tokens> <args>:            最大tokens容量" << std::endl;
    std::cout << "<--chunk_size> <args>:        Chunked Prefill分块大小 (默认: 自动)" << std::endl;
    std::cout << "<--model_name> <args>:        模型名(openai api中使用)" << std::endl;
    std::cout << "<--host> <args>:              监听地址 (默认: " << DEFAULT_API_HOST << ")" << std::endl;
    std::cout << "<--port> <args>:              网页端口号" << std::endl;
    std::cout << "<--cuda_embedding>:           使用cuda来执行embedding" << std::endl;
    std::cout << "<--device> <dev>:              执行设备 (如: cuda, cpu)" << std::endl;
    std::cout << "<--device_map> <map>:          设备分层映射 (如: cuda:28,cpu:8 表示28层GPU+8层CPU)" << std::endl;
    std::cout << "<--moe_device> <dev>:          MoE专家层设备" << std::endl;
    std::cout << "<--moe_device_map> <map>:      MoE专家层设备分层映射" << std::endl;
    std::cout << "<--api_key> <args>:           API Key (可选，设置后需要Bearer认证)" << std::endl;
    std::cout << "<--dev_mode>:                 开发模式 (启用调试接口 /v1/cancel, /v1/active_conversations)" << std::endl;
}

void ParseArgs(int argc, char **argv, APIConfig &config) {
    std::vector<std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        } else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "--embedding_path") {
            if (i + 1 >= argc) {
                Usage();
                exit(-1);
            }
            config.embeddingPath = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--low") {
            config.lowMemMode = true;
        } else if (sargv[i] == "--cuda_embedding"){
            config.cudaEmbedding = true;
        } else if (sargv[i] == "--host") {
            if (i + 1 >= argc) {
                Usage();
                exit(-1);
            }
            config.host = sargv[++i];
        } else if (sargv[i] == "--port") {
            config.port = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--dtype") {
            std::string dtypeStr = sargv[++i];
            if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
                config.groupCnt = atoi(dtypeStr.substr(5).c_str());
                dtypeStr = dtypeStr.substr(0, 5);
            }
            fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                                    "Unsupport data type: " + dtypeStr);
            config.dtype = dataTypeDict[dtypeStr];
        } else if (sargv[i] == "--tokens") {
            config.tokens = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--batch" || sargv[i] == "--max_batch") {
            config.batch = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--chunk_size" || sargv[i] == "--chunked_prefill_size") {
            config.chunkedPrefillSize = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--atype") {
            std::string atypeStr = sargv[++i];
            fastllm::AssertInFastLLM(dataTypeDict.find(atypeStr) != dataTypeDict.end(),
                                    "Unsupport act type: " + atypeStr);
            config.atype = dataTypeDict[atypeStr];
        } else if (sargv[i] == "--model_name") {
            config.modelName = sargv[++i];
        } else if (sargv[i] == "--device" || sargv[i] == "--device_map") {
            // 支持多种格式:
            // 1. 简单设备名: cpu, cuda
            // 2. 逗号分隔格式: cuda:28,cpu:8 或 cuda:0:28,cuda:1:8,cpu:4
            // 3. Python 字典格式: {'cuda':1,'cpu':4} 或 {"cuda":1,"cpu":4}
            std::string mapStr = sargv[++i];
            
            // 检查是否是 Python 字典格式
            if (mapStr.front() == '{' && mapStr.back() == '}') {
                mapStr = mapStr.substr(1, mapStr.length() - 2);
                std::string cleanStr;
                for (char c : mapStr) {
                    if (c != '\'' && c != '"') {
                        cleanStr += c;
                    }
                }
                mapStr = cleanStr;
            }
            
            std::stringstream ss(mapStr);
            std::string item;
            bool hasMapping = false;
            while (std::getline(ss, item, ',')) {
                size_t pos = item.rfind(':');
                if (pos != std::string::npos && pos > 0) {
                    std::string dev = item.substr(0, pos);
                    int layers = atoi(item.substr(pos + 1).c_str());
                    if (layers > 0) {
                        config.devices[dev] = layers;
                        hasMapping = true;
                    }
                }
            }
            if (!hasMapping && !mapStr.empty()) {
                config.devices[mapStr] = 1;
            }
        } else if (sargv[i] == "--moe_device" || sargv[i] == "--moe_device_map") {
            // 支持多种格式:
            // 1. 简单设备名: cpu, cuda
            // 2. 逗号分隔格式: cuda:1,cpu:4
            // 3. Python 字典格式: {'cuda':1,'cpu':4} 或 {"cuda":1,"cpu":4}
            std::string mapStr = sargv[++i];
            
            // 检查是否是 Python 字典格式
            if (mapStr.front() == '{' && mapStr.back() == '}') {
                // 去掉花括号
                mapStr = mapStr.substr(1, mapStr.length() - 2);
                // 替换单引号为空，处理 'cuda' -> cuda
                std::string cleanStr;
                for (char c : mapStr) {
                    if (c != '\'' && c != '"') {
                        cleanStr += c;
                    }
                }
                mapStr = cleanStr;
            }
            
            // 解析格式: cuda:1,cpu:4
            std::stringstream ss(mapStr);
            std::string item;
            bool hasMapping = false;
            while (std::getline(ss, item, ',')) {
                size_t pos = item.rfind(':');
                if (pos != std::string::npos && pos > 0) {
                    std::string dev = item.substr(0, pos);
                    int layers = atoi(item.substr(pos + 1).c_str());
                    if (layers > 0) {
                        config.moeDevices[dev] = layers;
                        hasMapping = true;
                    }
                }
            }
            // 如果没有解析出任何映射，当作简单设备名处理
            if (!hasMapping && !mapStr.empty()) {
                config.moeDevices[mapStr] = 1;
            }
        } else if (sargv[i] == "--api_key") {
            if (i + 1 >= argc) {
                Usage();
                exit(-1);
            }
            config.apiKey = sargv[++i];
        } else if (sargv[i] == "--dev_mode") {
            config.devMode = true;
        } else {
            Usage();
            exit(-1);
        }
    }
}

char buff[1024 * 1024] = {0};
std::string url = "generate";
std::mutex locker;

int main(int argc, char** argv) {
#ifdef _WIN32
    // Windows socket 初始化
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed!" << std::endl;
        return -1;
    }
#endif

    console::init();  // 初始化控制台美化
    ParseArgs(argc, argv, config);

    // 启用美化日志输出（一行代码即可）
    log_handler::EnablePrettyLogging();

    // 显示系统信息
    console::printHeader("系统信息");
    auto cpuFlags = fastllm::cpuInstructInfo.getFlags();
    std::string enabledFlags;
    if (cpuFlags.avx2) enabledFlags += "AVX2 ";
    if (cpuFlags.avx512f) enabledFlags += "AVX512F ";
    if (cpuFlags.avx512vnni) enabledFlags += "AVX512_VNNI ";
    if (cpuFlags.avx512bf16) enabledFlags += "AVX512_BF16 ";
    if (cpuFlags.amx) enabledFlags += "AMX ";
    if (enabledFlags.empty()) enabledFlags = "无";
    console::printConfig("CPU 指令集", enabledFlags);
    if (config.threads > 0) {
        console::printConfig("线程数", std::to_string(config.threads));
    }
    console::printConfig("低内存模式", config.lowMemMode ? "是" : "否");
    
    // 硬件配置（设备映射）
    if (config.devices.size() != 0) {
        fastllm::SetDeviceMap(config.devices);
        std::string deviceMapStr;
        for (auto &it : config.devices) {
            if (!deviceMapStr.empty()) deviceMapStr += ", ";
            deviceMapStr += it.first + ":" + std::to_string(it.second);
        }
        console::printConfig("设备映射", deviceMapStr);
    }
    if (config.moeDevices.size() != 0) {
        fastllm::SetMoeDeviceMap(config.moeDevices);
        std::string moeDeviceMapStr;
        for (auto &it : config.moeDevices) {
            if (!moeDeviceMapStr.empty()) moeDeviceMapStr += ", ";
            moeDeviceMapStr += it.first + ":" + std::to_string(it.second);
        }
        console::printConfig("MoE 设备映射", moeDeviceMapStr);
    }
    if (config.cudaEmbedding) {
        console::printConfig("CUDA Embedding", "是");
    }

    console::printHeader("API Server 配置");
    
    // 【模型配置】
    console::printInfo("【模型配置】");
    console::printConfig("模型路径", config.path);
    if (config.dtype != fastllm::DataType::FLOAT32) {
        std::string dtypeStr;
        switch (config.dtype) {
            case fastllm::DataType::FLOAT16: dtypeStr = "float16"; break;
            case fastllm::DataType::INT8: dtypeStr = "int8"; break;
            case fastllm::DataType::INT4: dtypeStr = "int4"; break;
            case fastllm::DataType::INT4_NOZERO: dtypeStr = "int4g"; break;
            default: dtypeStr = "auto";
        }
        console::printConfig("数据类型", dtypeStr);
    }
    if (config.atype != fastllm::DataType::FLOAT32) {
        std::string atypeStr = config.atype == fastllm::DataType::FLOAT16 ? "float16" : "float32";
        console::printConfig("激活类型", atypeStr);
    }
    if (config.tokens > 0) {
        console::printConfig("上下文限制", std::to_string(config.tokens) + " tokens");
    }
    if (config.chunkedPrefillSize > 0) {
        console::printConfig("分块 Prefill", std::to_string(config.chunkedPrefillSize));
    }
    
    // 【服务配置】
    console::printInfo("【服务配置】");
    console::printConfig("监听地址", config.host);
    console::printConfig("端口", std::to_string(config.port));
    if (config.batch > 1) {
        console::printConfig("最大批次", std::to_string(config.batch));
    }
    if (!config.apiKey.empty()) {
        console::printConfig("API Key", "******");
    }
    if (config.devMode) {
        console::printConfig("开发模式", "已启用");
    }
    
    fastllm::SetThreads(config.threads);
    fastllm::SetLowMemMode(config.lowMemMode);
    fastllm::SetCudaEmbedding(config.cudaEmbedding);
    
    if (!fastllm::FileExists(config.path)) {
        std::cerr << "模型文件 " << config.path << " 不存在！" << std::endl;
        exit(0);
    }
    bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
    workQueue.model = isHFDir ? fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt)
        : fastllm::CreateLLMModelFromFile(config.path);
    workQueue.model->SetSaveHistoryChat(true);

    if (!config.embeddingPath.empty()) {
        if (!fastllm::FileExists(config.embeddingPath)) {
            std::cerr << "embedding模型文件 " << config.embeddingPath << " 不存在！" << std::endl;
            exit(0);
        }
        workQueue.embeddingModel = fastllm::CreateEmbeddingModelFromFile(config.embeddingPath);
        workQueue.embeddingModel->SetSaveHistoryChat(false);
        workQueue.embeddingModel->SetDataType(config.atype);
        console::printConfig("Embedding 模型", config.embeddingPath);
    }

    workQueue.model->tokensLimit = config.tokens;
    workQueue.model->chunkedPrefillSize = config.chunkedPrefillSize;
    workQueue.model->SetDataType(config.atype);
    workQueue.model->verbose = true;  // 启用速度显示
    workQueue.maxActivateQueryNumber = std::max(1, std::min(256, config.batch));
    workQueue.Start();

    console::printHeader("网络初始化");

    socket_t local_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (local_fd == INVALID_SOCKET) {
        std::cerr << "socket error!" << std::endl;
        exit(-1);
    }
    console::printSuccess("Socket 已创建");

    std::string bindHost = config.host;
    if (bindHost == "localhost") {
        bindHost = DEFAULT_API_HOST;
    }

    in_addr bindInAddr{};
    {
        // 1) 优先按 IPv4 字符串解析 (无歧义)
        const int ptonRet = inet_pton(AF_INET, bindHost.c_str(), &bindInAddr);
        if (ptonRet != 1) {
            // 2) 回退到 getaddrinfo：支持 hostname (如机器名/域名)
            addrinfo hints{};
            hints.ai_family = AF_INET;
            hints.ai_socktype = SOCK_STREAM;
            addrinfo *result = nullptr;
            const int gaiRet = getaddrinfo(bindHost.c_str(), nullptr, &hints, &result);
            if (gaiRet != 0 || result == nullptr || result->ai_addr == nullptr) {
                std::cout << "invalid host: " << bindHost << std::endl;
                exit(-1);
            }
            auto *sa = reinterpret_cast<sockaddr_in *>(result->ai_addr);
            bindInAddr = sa->sin_addr;
            freeaddrinfo(result);
        }
    }
    struct sockaddr_in local_addr;
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(config.port);  //绑定端口
    local_addr.sin_addr = bindInAddr;

    //3.bind()： 将一个网络地址与一个套接字绑定，此处将本地地址绑定到一个套接字上
    int res = bind(local_fd, (struct sockaddr *) &local_addr, sizeof(local_addr));
    if (res == -1) {
        std::cerr << "bind error!" << std::endl;
        exit(-1);
    }
    console::printSuccess("端口绑定成功: " + std::to_string(config.port));
    listen(local_fd, 2000);    
    
    console::printHeader("服务就绪");
    console::printInfo("监听地址: http://" + config.host + ":" + std::to_string(config.port));
    console::printInfo("API 端点: /v1/chat/completions, /v1/completions, /v1/embeddings");
    std::cout << std::endl;
    
    // 设置 Windows 结构化异常处理
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    
    while (true) { //循环接收客户端的请求
        //5.创建一个sockaddr_in结构体，用来存储客户机的地址
        struct sockaddr_in client_addr;
        socklen_t len = sizeof(client_addr);
        //6.accept()函数：阻塞运行，直到收到某一客户机的连接请求，并返回客户机的描述符
        socket_t client = accept(local_fd, (struct sockaddr *) &client_addr, &len);
        
        if (client == INVALID_SOCKET) {
            exit(-1);
        }

        int size = 0;
        while (true) {
#ifdef _WIN32
            int cur = recv(client, buff + size, sizeof(buff) - size, 0);
            if (cur == SOCKET_ERROR) {
                closesocket(client);
                break;
            }
#else
            int cur = read(client, buff + size, sizeof(buff) - size);
#endif
            if (cur <= 0) {
                // 连接关闭
#ifdef _WIN32
                closesocket(client);
#else
                close(client);
#endif
                break;
            }
            size += cur;
            if (httpChecker.IsValid(buff, size)) {
                break;
            }
        }
        
        if (size == 0) {
            continue;  // 跳过空请求
        }
        buff[size] = 0;

        while (workQueue.q.size() > workQueue.maxActivateQueryNumber) {
            fastllm::MySleep(0);
        }
        workQueue.Push(buff, client);
    }

    return 0;
}

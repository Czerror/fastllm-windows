/*
 * chat_handler.cpp - Chat Template and Tool Call Handler Implementation
 */

#include "utils/chat_handler.h"
#include <minja/chat-template.hpp>
#include <regex>
#include <sstream>
#include <random>
#include <chrono>
#include <iostream>
#include <map>
#include <algorithm>

namespace fastllm {

// ============================================================================
// Tool Choice Implementation
// ============================================================================

ToolChoice parseToolChoice(const json& value) {
    if (value.is_null()) {
        return ToolChoice::Auto;
    }
    if (value.is_string()) {
        std::string s = value.get<std::string>();
        if (s == "none") return ToolChoice::None;
        if (s == "required") return ToolChoice::Required;
        return ToolChoice::Auto;
    }
    if (value.is_object()) {
        // Specific tool requested = Required
        return ToolChoice::Required;
    }
    return ToolChoice::Auto;
}

const char* toolChoiceToString(ToolChoice choice) {
    switch (choice) {
        case ToolChoice::Auto: return "auto";
        case ToolChoice::Required: return "required";
        case ToolChoice::None: return "none";
        default: return "auto";
    }
}

// ============================================================================
// ChatMsg Implementation
// ============================================================================

json ChatMsg::toJson() const {
    json msg = {{"role", role}};
    
    if (!content.empty()) {
        msg["content"] = content;
    } else if (!content_parts.empty()) {
        json parts = json::array();
        for (const auto& part : content_parts) {
            parts.push_back({{"type", part.type}, {"text", part.text}});
        }
        msg["content"] = parts;
    } else if (!tool_calls.empty()) {
        msg["content"] = nullptr;
    } else {
        msg["content"] = "";
    }
    
    if (!reasoning_content.empty()) {
        msg["reasoning_content"] = reasoning_content;
    }
    if (!tool_name.empty()) {
        msg["name"] = tool_name;
    }
    if (!tool_call_id.empty()) {
        msg["tool_call_id"] = tool_call_id;
    }
    if (!tool_calls.empty()) {
        msg["tool_calls"] = ChatHandler::toolCallsToJson(tool_calls);
    }
    
    return msg;
}

ChatMsg ChatMsg::fromJson(const json& j) {
    ChatMsg msg;
    
    if (j.contains("role")) {
        msg.role = j["role"].get<std::string>();
    }
    
    if (j.contains("content")) {
        const auto& content = j["content"];
        if (content.is_string()) {
            msg.content = content.get<std::string>();
        } else if (content.is_array()) {
            for (const auto& part : content) {
                ChatMsgContentPart p;
                p.type = part.value("type", "text");
                p.text = part.value("text", "");
                msg.content_parts.push_back(p);
            }
        }
    }
    
    if (j.contains("reasoning_content")) {
        msg.reasoning_content = j["reasoning_content"].get<std::string>();
    }
    if (j.contains("name")) {
        msg.tool_name = j["name"].get<std::string>();
    }
    if (j.contains("tool_call_id")) {
        msg.tool_call_id = j["tool_call_id"].get<std::string>();
    }
    
    if (j.contains("tool_calls") && j["tool_calls"].is_array()) {
        for (const auto& tc : j["tool_calls"]) {
            ToolCall call;
            call.id = tc.value("id", "");
            if (tc.contains("function")) {
                call.name = tc["function"].value("name", "");
                if (tc["function"]["arguments"].is_string()) {
                    call.arguments = tc["function"]["arguments"].get<std::string>();
                } else {
                    call.arguments = tc["function"]["arguments"].dump();
                }
            }
            call.is_complete = true;
            msg.tool_calls.push_back(call);
        }
    }
    
    return msg;
}

// ============================================================================
// Streaming Diff Implementation
// ============================================================================

static std::string stringDiff(const std::string& last, const std::string& current) {
    if (last.empty()) return current;
    if (current.size() < last.size()) return "";  // Shouldn't happen normally
    if (current.substr(0, last.size()) != last) return "";  // Mismatch
    return current.substr(last.size());
}

std::vector<ChatMsgDiff> ChatMsgDiff::computeDiffs(const ChatMsg& msg_prev, const ChatMsg& msg_new) {
    std::vector<ChatMsgDiff> diffs;
    diffs.reserve(3 + msg_new.tool_calls.size());
    
    // Reasoning content diff
    if (msg_prev.reasoning_content != msg_new.reasoning_content) {
        ChatMsgDiff diff;
        diff.reasoning_content_delta = stringDiff(msg_prev.reasoning_content, msg_new.reasoning_content);
        if (!diff.reasoning_content_delta.empty()) {
            diffs.push_back(diff);
        }
    }
    
    // Content diff
    if (msg_prev.content != msg_new.content) {
        ChatMsgDiff diff;
        diff.content_delta = stringDiff(msg_prev.content, msg_new.content);
        if (!diff.content_delta.empty()) {
            diffs.push_back(diff);
        }
    }
    
    // Tool calls diffs
    if (msg_new.tool_calls.size() < msg_prev.tool_calls.size()) {
        // This shouldn't happen in normal streaming
        return diffs;
    }
    
    // Update last tool call if arguments changed
    if (!msg_prev.tool_calls.empty()) {
        size_t idx = msg_prev.tool_calls.size() - 1;
        const auto& prev_tc = msg_prev.tool_calls[idx];
        const auto& new_tc = msg_new.tool_calls[idx];
        
        if (prev_tc.name != new_tc.name) {
            // Shouldn't happen
        } else {
            std::string args_diff = stringDiff(prev_tc.arguments, new_tc.arguments);
            if (!args_diff.empty() || prev_tc.id != new_tc.id) {
                ChatMsgDiff diff;
                diff.tool_call_index = idx;
                if (prev_tc.id != new_tc.id) {
                    diff.tool_call_delta.id = new_tc.id;
                    diff.tool_call_delta.name = new_tc.name;
                }
                diff.tool_call_delta.arguments = args_diff;
                diffs.push_back(diff);
            }
        }
    }
    
    // New tool calls
    for (size_t idx = msg_prev.tool_calls.size(); idx < msg_new.tool_calls.size(); ++idx) {
        ChatMsgDiff diff;
        diff.tool_call_index = idx;
        diff.tool_call_delta = msg_new.tool_calls[idx];
        diffs.push_back(diff);
    }
    
    return diffs;
}

json ChatMsgDiff::toJsonDelta() const {
    json delta = json::object();
    
    if (!reasoning_content_delta.empty()) {
        delta["reasoning_content"] = reasoning_content_delta;
    }
    if (!content_delta.empty()) {
        delta["content"] = content_delta;
    }
    
    if (tool_call_index != std::string::npos) {
        json tc = json::object();
        tc["index"] = tool_call_index;
        
        if (!tool_call_delta.id.empty()) {
            tc["id"] = tool_call_delta.id;
            tc["type"] = "function";
        }
        
        json func = json::object();
        if (!tool_call_delta.name.empty()) {
            func["name"] = tool_call_delta.name;
        }
        func["arguments"] = tool_call_delta.arguments;
        tc["function"] = func;
        
        delta["tool_calls"] = json::array({tc});
    }
    
    return delta;
}

ChatMsg StreamingParseResult::toChatMsg() const {
    ChatMsg msg;
    msg.role = "assistant";
    msg.content = content;
    msg.reasoning_content = reasoning_content;
    msg.tool_calls = tool_calls;
    return msg;
}

// ============================================================================
// ChatHandler Implementation
// ============================================================================

class ChatHandler::Impl {
public:
    std::unique_ptr<minja::chat_template> template_;
    std::string bos_token_;
    std::string eos_token_;

    Impl(const std::string& source, const std::string& bos, const std::string& eos)
        : bos_token_(bos), eos_token_(eos)
    {
        if (!source.empty()) {
            try {
                template_ = std::make_unique<minja::chat_template>(source, bos, eos);
            } catch (const std::exception& e) {
                std::cerr << "[ChatHandler] Failed to parse template: " << e.what() << std::endl;
                template_.reset();
            }
        }
    }
};

ChatHandler::ChatHandler(const std::string& template_source,
                         const std::string& bos_token,
                         const std::string& eos_token)
    : impl_(std::make_unique<Impl>(template_source, bos_token, eos_token))
{
}

ChatHandler::~ChatHandler() = default;

bool ChatHandler::hasTemplate() const {
    return impl_->template_ != nullptr;
}

ChatTemplateCaps ChatHandler::getCapabilities() const {
    ChatTemplateCaps caps;
    if (impl_->template_) {
        const auto& orig = impl_->template_->original_caps();
        caps.supports_tools = orig.supports_tools;
        caps.supports_tool_calls = orig.supports_tool_calls;
        caps.supports_tool_responses = orig.supports_tool_responses;
        caps.supports_system_role = orig.supports_system_role;
        caps.supports_parallel_tool_calls = orig.supports_parallel_tool_calls;
        caps.requires_object_arguments = orig.requires_object_arguments;
    }
    return caps;
}

std::string ChatHandler::applyTemplate(
    const json& messages,
    const json& tools,
    bool add_generation_prompt,
    const json& extra_context) const
{
    if (!impl_->template_) {
        throw std::runtime_error("No chat template available");
    }

    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.tools = tools;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.extra_context = extra_context;
    inputs.now = std::chrono::system_clock::now();

    minja::chat_template_options opts;
    opts.apply_polyfills = true;

    return impl_->template_->apply(inputs, opts);
}

std::vector<ToolCall> ChatHandler::parseToolCalls(const std::string& output) const {
    StreamingToolCallParser parser;
    parser.feed(output);
    return parser.finalize().tool_calls;
}

std::function<StreamingParseResult(const std::string&)> ChatHandler::createStreamingParser() const {
    auto parser = std::make_shared<StreamingToolCallParser>();
    return [parser](const std::string& chunk) {
        return parser->feed(chunk);
    };
}

json ChatHandler::toolCallsToJson(const std::vector<ToolCall>& tool_calls) {
    json result = json::array();
    for (const auto& tc : tool_calls) {
        json item = {
            {"id", tc.id},
            {"type", "function"},
            {"function", {
                {"name", tc.name},
                {"arguments", tc.arguments}
            }}
        };
        result.push_back(item);
    }
    return result;
}

json ChatHandler::buildAssistantMessage(
    const std::string& content,
    const std::string& reasoning_content,
    const std::vector<ToolCall>& tool_calls)
{
    json message = {
        {"role", "assistant"}
    };
    
    // Add reasoning_content if present (OpenAI extended format)
    if (!reasoning_content.empty()) {
        message["reasoning_content"] = reasoning_content;
    }
    
    // Handle content and tool_calls
    if (tool_calls.empty()) {
        message["content"] = content;
    } else {
        // When tool calls present, content can be null or contain thinking
        if (content.empty()) {
            message["content"] = nullptr;
        } else {
            message["content"] = content;
        }
        message["tool_calls"] = toolCallsToJson(tool_calls);
    }
    
    return message;
}

std::string ChatHandler::generateToolCallId(int index) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 35);
    
    std::string id = "call_";
    const char charset[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    for (int i = 0; i < 24; ++i) {
        id += charset[dis(gen)];
    }
    if (index >= 0) {
        id += "_" + std::to_string(index);
    }
    return id;
}

// ============================================================================
// StreamingToolCallParser Implementation
// ============================================================================

class StreamingToolCallParser::Impl {
public:
    std::string buffer_;
    std::vector<ToolCall> pending_tool_calls_;
    std::string content_buffer_;
    std::string reasoning_buffer_;   // Accumulated reasoning content
    bool in_tool_call_ = false;
    bool in_reasoning_ = false;      // Currently inside thinking tags
    bool thinking_forced_open_ = false;  // Thinking started but not closed
    int tool_call_index_ = 0;
    
    // Detection patterns
    enum class Format {
        Unknown,
        JsonBlock,      // ```json ... ```
        Qwen3,          // <tool_call>...</tool_call>
        DeepSeek,       // <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>
        LlamaStyle,     // <function=name>{...}</function>
        DirectJson      // Direct JSON object
    };
    Format detected_format_ = Format::Unknown;

    // Qwen3 markers
    const std::string QWEN3_START = "<tool_call>";
    const std::string QWEN3_END = "</tool_call>";
    
    // DeepSeek markers
    const std::string DS_START = "<｜tool▁calls▁begin｜>";
    const std::string DS_END = "<｜tool▁calls▁end｜>";
    const std::string DS_SEP = "<｜tool▁sep｜>";
    
    // JSON block markers
    const std::string JSON_BLOCK_START = "```json";
    const std::string JSON_BLOCK_END = "```";
    
    // Reasoning/Thinking markers (multiple formats)
    const std::vector<std::pair<std::string, std::string>> THINKING_MARKERS = {
        {"<think>", "</think>"},           // DeepSeek R1, Hermes, etc.
        {"<thinking>", "</thinking>"},     // Alternative format
        {"<｜thinking｜>", "<｜/thinking｜>"},  // DeepSeek alternative
    };

    StreamingParseResult processBuffer() {
        StreamingParseResult result;
        
        // First, process any reasoning/thinking content
        processReasoning(result);
        
        // Try to detect format if unknown
        if (detected_format_ == Format::Unknown) {
            detectFormat();
        }

        switch (detected_format_) {
            case Format::Qwen3:
                processQwen3Format(result);
                break;
            case Format::DeepSeek:
                processDeepSeekFormat(result);
                break;
            case Format::JsonBlock:
                processJsonBlockFormat(result);
                break;
            case Format::DirectJson:
                processDirectJsonFormat(result);
                break;
            default:
                // No tool call detected yet, accumulate as content
                result.content += buffer_;
                buffer_.clear();
                break;
        }
        
        // Set reasoning info
        if (!reasoning_buffer_.empty()) {
            result.reasoning_content = reasoning_buffer_;
            result.has_reasoning = true;
        }
        result.thinking_forced_open = thinking_forced_open_;

        return result;
    }
    
    void processReasoning(StreamingParseResult& result) {
        for (const auto& [start_tag, end_tag] : THINKING_MARKERS) {
            size_t start_pos = buffer_.find(start_tag);
            if (start_pos != std::string::npos) {
                // Found thinking start tag
                // Content before thinking
                if (start_pos > 0) {
                    content_buffer_ += buffer_.substr(0, start_pos);
                }
                
                size_t end_pos = buffer_.find(end_tag, start_pos + start_tag.length());
                if (end_pos != std::string::npos) {
                    // Complete thinking block
                    std::string thinking = buffer_.substr(
                        start_pos + start_tag.length(),
                        end_pos - start_pos - start_tag.length()
                    );
                    reasoning_buffer_ += thinking;
                    buffer_ = buffer_.substr(end_pos + end_tag.length());
                    in_reasoning_ = false;
                    thinking_forced_open_ = false;
                } else {
                    // Incomplete thinking block
                    reasoning_buffer_ += buffer_.substr(start_pos + start_tag.length());
                    buffer_.clear();
                    in_reasoning_ = true;
                    thinking_forced_open_ = true;
                }
                return;
            }
            
            // Check if we're continuing inside thinking
            if (in_reasoning_) {
                size_t end_pos = buffer_.find(end_tag);
                if (end_pos != std::string::npos) {
                    reasoning_buffer_ += buffer_.substr(0, end_pos);
                    buffer_ = buffer_.substr(end_pos + end_tag.length());
                    in_reasoning_ = false;
                    thinking_forced_open_ = false;
                } else {
                    reasoning_buffer_ += buffer_;
                    buffer_.clear();
                }
                return;
            }
        }
    }

    void detectFormat() {
        if (buffer_.find(QWEN3_START) != std::string::npos) {
            detected_format_ = Format::Qwen3;
        } else if (buffer_.find(DS_START) != std::string::npos) {
            detected_format_ = Format::DeepSeek;
        } else if (buffer_.find(JSON_BLOCK_START) != std::string::npos) {
            detected_format_ = Format::JsonBlock;
        } else if (buffer_.find("{\"name\"") != std::string::npos || 
                   buffer_.find("{ \"name\"") != std::string::npos) {
            // Check if it looks like a tool call JSON
            if (buffer_.find("\"arguments\"") != std::string::npos) {
                detected_format_ = Format::DirectJson;
            }
        }
    }

    void processQwen3Format(StreamingParseResult& result) {
        size_t start_pos;
        while ((start_pos = buffer_.find(QWEN3_START)) != std::string::npos) {
            // Content before tool call
            if (start_pos > 0) {
                result.content += buffer_.substr(0, start_pos);
            }
            
            size_t end_pos = buffer_.find(QWEN3_END, start_pos);
            if (end_pos == std::string::npos) {
                // Incomplete tool call, wait for more data
                buffer_ = buffer_.substr(start_pos);
                in_tool_call_ = true;
                return;
            }
            
            // Extract tool call content
            size_t content_start = start_pos + QWEN3_START.length();
            std::string tool_content = buffer_.substr(content_start, end_pos - content_start);
            
            // Parse the tool call JSON
            ToolCall tc = parseToolCallJson(tool_content);
            if (!tc.name.empty()) {
                tc.is_complete = true;
                result.tool_calls.push_back(tc);
                result.has_tool_calls = true;
            }
            
            buffer_ = buffer_.substr(end_pos + QWEN3_END.length());
            in_tool_call_ = false;
        }
        
        // Remaining buffer might be content or partial tool call
        if (!in_tool_call_ && !buffer_.empty()) {
            // Check if buffer might start a new tool call
            if (buffer_.find("<tool") == std::string::npos) {
                result.content += buffer_;
                buffer_.clear();
            }
        }
    }

    void processDeepSeekFormat(StreamingParseResult& result) {
        size_t start_pos = buffer_.find(DS_START);
        if (start_pos == std::string::npos) {
            result.content += buffer_;
            buffer_.clear();
            return;
        }
        
        // Content before tool calls
        if (start_pos > 0) {
            result.content += buffer_.substr(0, start_pos);
        }
        
        size_t end_pos = buffer_.find(DS_END, start_pos);
        if (end_pos == std::string::npos) {
            // Incomplete, wait for more
            buffer_ = buffer_.substr(start_pos);
            in_tool_call_ = true;
            return;
        }
        
        // Extract all tool calls
        size_t content_start = start_pos + DS_START.length();
        std::string tools_content = buffer_.substr(content_start, end_pos - content_start);
        
        // Split by separator if present
        std::vector<std::string> tool_strings;
        size_t sep_pos;
        while ((sep_pos = tools_content.find(DS_SEP)) != std::string::npos) {
            tool_strings.push_back(tools_content.substr(0, sep_pos));
            tools_content = tools_content.substr(sep_pos + DS_SEP.length());
        }
        if (!tools_content.empty()) {
            tool_strings.push_back(tools_content);
        }
        
        // Parse each tool call
        for (const auto& ts : tool_strings) {
            ToolCall tc = parseToolCallJson(ts);
            if (!tc.name.empty()) {
                tc.is_complete = true;
                result.tool_calls.push_back(tc);
                result.has_tool_calls = true;
            }
        }
        
        buffer_ = buffer_.substr(end_pos + DS_END.length());
        in_tool_call_ = false;
    }

    void processJsonBlockFormat(StreamingParseResult& result) {
        size_t start_pos = buffer_.find(JSON_BLOCK_START);
        if (start_pos == std::string::npos) {
            result.content += buffer_;
            buffer_.clear();
            return;
        }
        
        // Content before JSON block
        if (start_pos > 0) {
            result.content += buffer_.substr(0, start_pos);
        }
        
        size_t content_start = start_pos + JSON_BLOCK_START.length();
        // Skip newline after ```json
        if (content_start < buffer_.size() && buffer_[content_start] == '\n') {
            content_start++;
        }
        
        size_t end_pos = buffer_.find(JSON_BLOCK_END, content_start);
        if (end_pos == std::string::npos) {
            buffer_ = buffer_.substr(start_pos);
            in_tool_call_ = true;
            return;
        }
        
        std::string json_content = buffer_.substr(content_start, end_pos - content_start);
        
        // Try to parse as tool call
        ToolCall tc = parseToolCallJson(json_content);
        if (!tc.name.empty()) {
            tc.is_complete = true;
            result.tool_calls.push_back(tc);
            result.has_tool_calls = true;
        }
        
        buffer_ = buffer_.substr(end_pos + JSON_BLOCK_END.length());
        in_tool_call_ = false;
    }

    void processDirectJsonFormat(StreamingParseResult& result) {
        // Find JSON object boundaries
        size_t obj_start = buffer_.find('{');
        if (obj_start == std::string::npos) {
            result.content += buffer_;
            buffer_.clear();
            return;
        }
        
        // Content before JSON
        if (obj_start > 0) {
            result.content += buffer_.substr(0, obj_start);
        }
        
        // Find matching closing brace
        int depth = 0;
        size_t obj_end = std::string::npos;
        for (size_t i = obj_start; i < buffer_.size(); ++i) {
            if (buffer_[i] == '{') depth++;
            else if (buffer_[i] == '}') {
                depth--;
                if (depth == 0) {
                    obj_end = i;
                    break;
                }
            }
        }
        
        if (obj_end == std::string::npos) {
            // Incomplete JSON
            buffer_ = buffer_.substr(obj_start);
            in_tool_call_ = true;
            return;
        }
        
        std::string json_str = buffer_.substr(obj_start, obj_end - obj_start + 1);
        ToolCall tc = parseToolCallJson(json_str);
        if (!tc.name.empty()) {
            tc.is_complete = true;
            result.tool_calls.push_back(tc);
            result.has_tool_calls = true;
        }
        
        buffer_ = buffer_.substr(obj_end + 1);
        in_tool_call_ = false;
    }

    ToolCall parseToolCallJson(const std::string& json_str) {
        ToolCall tc;
        try {
            // Trim whitespace
            std::string trimmed = json_str;
            size_t start = trimmed.find_first_not_of(" \t\n\r");
            size_t end = trimmed.find_last_not_of(" \t\n\r");
            if (start != std::string::npos && end != std::string::npos) {
                trimmed = trimmed.substr(start, end - start + 1);
            }
            
            if (trimmed.empty() || trimmed[0] != '{') {
                return tc;
            }
            
            auto j = json::parse(trimmed);
            
            // Extract name
            if (j.contains("name")) {
                tc.name = j["name"].get<std::string>();
            } else if (j.contains("function") && j["function"].contains("name")) {
                tc.name = j["function"]["name"].get<std::string>();
            }
            
            // Extract arguments
            if (j.contains("arguments")) {
                if (j["arguments"].is_string()) {
                    tc.arguments = j["arguments"].get<std::string>();
                } else {
                    tc.arguments = j["arguments"].dump();
                }
            } else if (j.contains("function") && j["function"].contains("arguments")) {
                auto& args = j["function"]["arguments"];
                if (args.is_string()) {
                    tc.arguments = args.get<std::string>();
                } else {
                    tc.arguments = args.dump();
                }
            } else if (j.contains("parameters")) {
                // Alternative format
                tc.arguments = j["parameters"].dump();
            }
            
            // Extract or generate ID
            if (j.contains("id")) {
                tc.id = j["id"].get<std::string>();
            } else {
                tc.id = ChatHandler::generateToolCallId(tool_call_index_++);
            }
            
        } catch (const std::exception& e) {
            // JSON parsing failed, return empty tool call
            tc = ToolCall();
        }
        
        return tc;
    }
};

StreamingToolCallParser::StreamingToolCallParser()
    : impl_(std::make_unique<Impl>())
{
}

StreamingToolCallParser::~StreamingToolCallParser() = default;

StreamingParseResult StreamingToolCallParser::feed(const std::string& chunk) {
    impl_->buffer_ += chunk;
    return impl_->processBuffer();
}

StreamingParseResult StreamingToolCallParser::finalize() {
    StreamingParseResult result = impl_->processBuffer();
    result.parsing_complete = true;
    
    // If we have remaining buffer, treat as content
    if (!impl_->buffer_.empty()) {
        result.content += impl_->buffer_;
        impl_->buffer_.clear();
    }
    
    return result;
}

void StreamingToolCallParser::reset() {
    impl_->buffer_.clear();
    impl_->pending_tool_calls_.clear();
    impl_->content_buffer_.clear();
    impl_->reasoning_buffer_.clear();
    impl_->in_tool_call_ = false;
    impl_->in_reasoning_ = false;
    impl_->thinking_forced_open_ = false;
    impl_->tool_call_index_ = 0;
    impl_->detected_format_ = Impl::Format::Unknown;
}

bool StreamingToolCallParser::isParsingToolCall() const {
    return impl_->in_tool_call_;
}

ChatMsg StreamingToolCallParser::getCurrentMessage() const {
    ChatMsg msg;
    msg.role = "assistant";
    msg.content = impl_->content_buffer_;
    msg.reasoning_content = impl_->reasoning_buffer_;
    msg.tool_calls = impl_->pending_tool_calls_;
    return msg;
}

std::vector<ChatMsgDiff> StreamingToolCallParser::computeDiffSinceLastCall(const ChatMsg& last_msg) const {
    return ChatMsgDiff::computeDiffs(last_msg, getCurrentMessage());
}

// ============================================================================
// StreamingDiffTracker Implementation
// ============================================================================

class StreamingDiffTracker::Impl {
public:
    ChatMsg last_msg_;
    bool first_update_ = true;
    
    Impl() {
        last_msg_.role = "assistant";
    }
};

StreamingDiffTracker::StreamingDiffTracker()
    : impl_(std::make_unique<Impl>())
{
}

StreamingDiffTracker::~StreamingDiffTracker() = default;

std::vector<ChatMsgDiff> StreamingDiffTracker::update(const ChatMsg& msg) {
    if (impl_->first_update_) {
        impl_->first_update_ = false;
        ChatMsg empty;
        empty.role = "assistant";
        auto diffs = ChatMsgDiff::computeDiffs(empty, msg);
        impl_->last_msg_ = msg;
        return diffs;
    }
    
    auto diffs = ChatMsgDiff::computeDiffs(impl_->last_msg_, msg);
    impl_->last_msg_ = msg;
    return diffs;
}

std::vector<ChatMsgDiff> StreamingDiffTracker::update(const StreamingParseResult& result) {
    ChatMsg msg = result.toChatMsg();
    return update(msg);
}

void StreamingDiffTracker::reset() {
    impl_->last_msg_ = ChatMsg();
    impl_->last_msg_.role = "assistant";
    impl_->first_update_ = true;
}

const ChatMsg& StreamingDiffTracker::getCurrentMessage() const {
    return impl_->last_msg_;
}

// ============================================================================
// PEG Parser Implementation
// ============================================================================

namespace {

// Helper function to cast PegRule to internal PegRuleBase
class PegRuleBase;
static PegParseResult parseRule(const std::shared_ptr<PegRule>& rule, const std::string& input, size_t pos);

// Base class for all rules (internal implementation)
class PegRuleBase : public PegRule {
public:
    virtual ~PegRuleBase() = default;
    virtual PegParseResult parse(const std::string& input, size_t pos) const = 0;
    
    // Interface methods
    std::string getName() const override { return name_; }
    void setName(const std::string& name) { name_ = name; }

protected:
    std::string name_;
};

// Helper function implementation
static PegParseResult parseRule(const std::shared_ptr<PegRule>& rule, const std::string& input, size_t pos) {
    auto base = std::dynamic_pointer_cast<PegRuleBase>(rule);
    if (base) {
        return base->parse(input, pos);
    }
    PegParseResult result;
    result.success = false;
    return result;
}

// Literal matching rule
class LiteralRule : public PegRuleBase {
public:
    explicit LiteralRule(const std::string& text, bool ignore_case = false)
        : text_(text), ignore_case_(ignore_case) {}

    PegParseResult parse(const std::string& input, size_t pos) const override {
        PegParseResult result;
        result.start_pos = pos;
        
        if (pos + text_.size() > input.size()) {
            result.success = false;
            return result;
        }
        
        bool match = true;
        for (size_t i = 0; i < text_.size() && match; ++i) {
            char a = input[pos + i];
            char b = text_[i];
            if (ignore_case_) {
                a = std::tolower(a);
                b = std::tolower(b);
            }
            if (a != b) match = false;
        }
        
        if (match) {
            result.success = true;
            result.matched = text_;
            result.end_pos = pos + text_.size();
        }
        
        return result;
    }

private:
    std::string text_;
    bool ignore_case_;
};

// Until rule - matches until delimiter
class UntilRule : public PegRuleBase {
public:
    UntilRule(const std::string& delimiter, bool include_delimiter)
        : delimiter_(delimiter), include_delimiter_(include_delimiter) {}

    PegParseResult parse(const std::string& input, size_t pos) const override {
        PegParseResult result;
        result.start_pos = pos;
        
        size_t found = input.find(delimiter_, pos);
        if (found == std::string::npos) {
            result.success = false;
            return result;
        }
        
        result.success = true;
        result.matched = input.substr(pos, found - pos);
        result.end_pos = found + (include_delimiter_ ? delimiter_.size() : 0);
        
        return result;
    }

private:
    std::string delimiter_;
    bool include_delimiter_;
};

// Regex rule
class RegexRule : public PegRuleBase {
public:
    explicit RegexRule(const std::string& pattern)
        : pattern_(pattern, std::regex::ECMAScript) {}

    PegParseResult parse(const std::string& input, size_t pos) const override {
        PegParseResult result;
        result.start_pos = pos;
        
        std::smatch match;
        std::string sub = input.substr(pos);
        if (std::regex_search(sub, match, pattern_, std::regex_constants::match_continuous)) {
            result.success = true;
            result.matched = match[0].str();
            result.end_pos = pos + match[0].length();
        }
        
        return result;
    }

private:
    std::regex pattern_;
};

// Sequence rule
class SequenceRule : public PegRuleBase {
public:
    explicit SequenceRule(std::vector<std::shared_ptr<PegRule>> rules) : rules_(std::move(rules)) {}

    PegParseResult parse(const std::string& input, size_t pos) const override {
        PegParseResult result;
        result.start_pos = pos;
        
        size_t current_pos = pos;
        std::string matched;
        
        for (const auto& rule : rules_) {
            auto r = parseRule(rule, input, current_pos);
            if (!r.success) {
                result.success = false;
                return result;
            }
            matched += r.matched;
            current_pos = r.end_pos;
            
            // Merge captures
            for (const auto& [k, v] : r.captures) {
                result.captures[k] = v;
            }
        }
        
        result.success = true;
        result.matched = matched;
        result.end_pos = current_pos;
        
        return result;
    }

private:
    std::vector<std::shared_ptr<PegRule>> rules_;
};

// Choice rule
class ChoiceRule : public PegRuleBase {
public:
    explicit ChoiceRule(std::vector<std::shared_ptr<PegRule>> rules) : rules_(std::move(rules)) {}

    PegParseResult parse(const std::string& input, size_t pos) const override {
        for (const auto& rule : rules_) {
            auto r = parseRule(rule, input, pos);
            if (r.success) {
                return r;
            }
        }
        
        PegParseResult result;
        result.start_pos = pos;
        result.success = false;
        return result;
    }

private:
    std::vector<std::shared_ptr<PegRule>> rules_;
};

// Optional rule
class OptionalRule : public PegRuleBase {
public:
    explicit OptionalRule(std::shared_ptr<PegRule> rule) : rule_(std::move(rule)) {}

    PegParseResult parse(const std::string& input, size_t pos) const override {
        auto r = parseRule(rule_, input, pos);
        if (r.success) {
            return r;
        }
        
        PegParseResult result;
        result.success = true;
        result.start_pos = pos;
        result.end_pos = pos;
        return result;
    }

private:
    std::shared_ptr<PegRule> rule_;
};

// Repeat rule (zero or more)
class RepeatRule : public PegRuleBase {
public:
    RepeatRule(std::shared_ptr<PegRule> rule, size_t min_count, size_t max_count)
        : rule_(std::move(rule)), min_count_(min_count), max_count_(max_count) {}

    PegParseResult parse(const std::string& input, size_t pos) const override {
        PegParseResult result;
        result.start_pos = pos;
        
        size_t current_pos = pos;
        size_t count = 0;
        std::string matched;
        
        while (count < max_count_) {
            auto r = parseRule(rule_, input, current_pos);
            if (!r.success || r.end_pos == current_pos) break;
            
            matched += r.matched;
            current_pos = r.end_pos;
            ++count;
            
            for (const auto& [k, v] : r.captures) {
                result.captures[k] = v;
            }
        }
        
        if (count < min_count_) {
            result.success = false;
            return result;
        }
        
        result.success = true;
        result.matched = matched;
        result.end_pos = current_pos;
        
        return result;
    }

private:
    std::shared_ptr<PegRule> rule_;
    size_t min_count_;
    size_t max_count_;
};

// Capture rule
class CaptureRule : public PegRuleBase {
public:
    CaptureRule(const std::string& name, std::shared_ptr<PegRule> rule)
        : capture_name_(name), rule_(std::move(rule)) {}

    PegParseResult parse(const std::string& input, size_t pos) const override {
        auto r = parseRule(rule_, input, pos);
        if (r.success) {
            r.captures[capture_name_] = r.matched;
        }
        return r;
    }

private:
    std::string capture_name_;
    std::shared_ptr<PegRule> rule_;
};

// Any char rule
class AnyCharRule : public PegRuleBase {
public:
    explicit AnyCharRule(size_t count = 1) : count_(count) {}

    PegParseResult parse(const std::string& input, size_t pos) const override {
        PegParseResult result;
        result.start_pos = pos;
        
        if (pos + count_ <= input.size()) {
            result.success = true;
            result.matched = input.substr(pos, count_);
            result.end_pos = pos + count_;
        }
        
        return result;
    }

private:
    size_t count_;
};

} // anonymous namespace

// PegParser Implementation
class PegParser::Impl {
public:
    std::shared_ptr<PegRule> root_;
};

PegParser::PegParser() : impl_(std::make_unique<Impl>()) {}
PegParser::~PegParser() = default;

std::shared_ptr<PegRule> PegParser::literal(const std::string& text) {
    return std::make_shared<LiteralRule>(text);
}

std::shared_ptr<PegRule> PegParser::literalIgnoreCase(const std::string& text) {
    return std::make_shared<LiteralRule>(text, true);
}

std::shared_ptr<PegRule> PegParser::until(const std::string& delimiter, bool include) {
    return std::make_shared<UntilRule>(delimiter, include);
}

std::shared_ptr<PegRule> PegParser::regex(const std::string& pattern) {
    return std::make_shared<RegexRule>(pattern);
}

std::shared_ptr<PegRule> PegParser::sequence(std::initializer_list<std::shared_ptr<PegRule>> rules) {
    std::vector<std::shared_ptr<PegRule>> vec(rules);
    return std::make_shared<SequenceRule>(std::move(vec));
}

std::shared_ptr<PegRule> PegParser::choice(std::initializer_list<std::shared_ptr<PegRule>> rules) {
    std::vector<std::shared_ptr<PegRule>> vec(rules);
    return std::make_shared<ChoiceRule>(std::move(vec));
}

std::shared_ptr<PegRule> PegParser::optional(std::shared_ptr<PegRule> rule) {
    return std::make_shared<OptionalRule>(std::move(rule));
}

std::shared_ptr<PegRule> PegParser::zeroOrMore(std::shared_ptr<PegRule> rule) {
    return std::make_shared<RepeatRule>(std::move(rule), 0, SIZE_MAX);
}

std::shared_ptr<PegRule> PegParser::oneOrMore(std::shared_ptr<PegRule> rule) {
    return std::make_shared<RepeatRule>(std::move(rule), 1, SIZE_MAX);
}

std::shared_ptr<PegRule> PegParser::repeat(std::shared_ptr<PegRule> rule, size_t min_count, size_t max_count) {
    return std::make_shared<RepeatRule>(std::move(rule), min_count, max_count);
}

std::shared_ptr<PegRule> PegParser::capture(const std::string& name, std::shared_ptr<PegRule> rule) {
    return std::make_shared<CaptureRule>(name, std::move(rule));
}

std::shared_ptr<PegRule> PegParser::anyChar(size_t count) {
    return std::make_shared<AnyCharRule>(count);
}

void PegParser::setRoot(std::shared_ptr<PegRule> rule) {
    impl_->root_ = std::move(rule);
}

PegParseResult PegParser::parse(const std::string& input, size_t start_pos) const {
    if (!impl_->root_) {
        PegParseResult result;
        result.success = false;
        return result;
    }
    return parseRule(impl_->root_, input, start_pos);
}

std::vector<PegParseResult> PegParser::parseAll(const std::string& input) const {
    std::vector<PegParseResult> results;
    if (!impl_->root_) return results;
    
    size_t pos = 0;
    while (pos < input.size()) {
        auto r = parseRule(impl_->root_, input, pos);
        if (r.success && r.end_pos > pos) {
            results.push_back(r);
            pos = r.end_pos;
        } else {
            ++pos;  // Skip one char and retry
        }
    }
    
    return results;
}

// ============================================================================
// Predefined PEG Parsers
// ============================================================================

namespace peg_parsers {

std::unique_ptr<PegParser> createQwen3Parser() {
    auto parser = std::make_unique<PegParser>();
    
    // Qwen3 format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    auto tool_call_rule = parser->sequence({
        parser->literal("<tool_call>"),
        parser->capture("json", parser->until("</tool_call>", false)),
        parser->literal("</tool_call>")
    });
    
    parser->setRoot(tool_call_rule);
    return parser;
}

std::unique_ptr<PegParser> createDeepSeekParser() {
    auto parser = std::make_unique<PegParser>();
    
    // DeepSeek format: <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>
    auto tool_call_rule = parser->sequence({
        parser->literal("<｜tool▁calls▁begin｜>"),
        parser->capture("json", parser->until("<｜tool▁calls▁end｜>", false)),
        parser->literal("<｜tool▁calls▁end｜>")
    });
    
    parser->setRoot(tool_call_rule);
    return parser;
}

std::unique_ptr<PegParser> createThinkingParser() {
    auto parser = std::make_unique<PegParser>();
    
    // Thinking tags: <think>...</think> or <thinking>...</thinking>
    auto think_rule = parser->choice({
        parser->sequence({
            parser->literal("<think>"),
            parser->capture("thinking", parser->until("</think>", false)),
            parser->literal("</think>")
        }),
        parser->sequence({
            parser->literal("<thinking>"),
            parser->capture("thinking", parser->until("</thinking>", false)),
            parser->literal("</thinking>")
        })
    });
    
    parser->setRoot(think_rule);
    return parser;
}

std::unique_ptr<PegParser> createJsonBlockParser() {
    auto parser = std::make_unique<PegParser>();
    
    // JSON code block: ```json\n...\n```
    auto json_block_rule = parser->sequence({
        parser->literal("```json"),
        parser->optional(parser->literal("\n")),
        parser->capture("json", parser->until("```", false)),
        parser->literal("```")
    });
    
    parser->setRoot(json_block_rule);
    return parser;
}

} // namespace peg_parsers

} // namespace fastllm

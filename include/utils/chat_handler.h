/*
 * chat_handler.h - Chat Template and Tool Call Handler
 * 
 * This module provides a unified interface for handling chat templates
 * and tool calls using the minja library.
 * 
 * Features:
 * - Jinja2 chat template rendering (via minja)
 * - Tool call parsing (streaming and non-streaming)
 * - Tool choice handling (auto/required/none)
 * - Streaming diff computation for incremental updates
 * - PEG-based output parsing for complex formats
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <nlohmann/json.hpp>

// Forward declaration for minja (in global namespace)
namespace minja { class chat_template; }

namespace fastllm {

using json = nlohmann::ordered_json;

// ============================================================================
// Tool Choice Handling
// ============================================================================

/**
 * Tool choice mode - controls when/if tools should be called
 */
enum class ToolChoice {
    Auto,      // Model decides whether to call tools (default)
    Required,  // Model must call at least one tool
    None       // Model should not call any tools
};

/**
 * Parse tool_choice from OpenAI API format
 * @param value String or object from API request
 * @return Parsed ToolChoice enum
 */
ToolChoice parseToolChoice(const json& value);

/**
 * Convert ToolChoice to string for display
 */
const char* toolChoiceToString(ToolChoice choice);

// ============================================================================
// Core Data Structures
// ============================================================================

/**
 * Tool call information extracted from model output
 */
struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // JSON string
    bool is_complete = false;
    
    bool operator==(const ToolCall& other) const {
        return id == other.id && name == other.name && arguments == other.arguments;
    }
    bool operator!=(const ToolCall& other) const {
        return !(*this == other);
    }
};

/**
 * Chat message content part (for multi-part content)
 */
struct ChatMsgContentPart {
    std::string type;  // "text", "image_url", etc.
    std::string text;
    
    bool operator==(const ChatMsgContentPart& other) const {
        return type == other.type && text == other.text;
    }
};

/**
 * Complete chat message structure (OpenAI compatible)
 */
struct ChatMsg {
    std::string role;
    std::string content;
    std::vector<ChatMsgContentPart> content_parts;
    std::vector<ToolCall> tool_calls;
    std::string reasoning_content;  // For thinking models
    std::string tool_name;          // For tool role
    std::string tool_call_id;       // For tool responses
    
    bool empty() const {
        return content.empty() && content_parts.empty() && 
               tool_calls.empty() && reasoning_content.empty();
    }
    
    bool operator==(const ChatMsg& other) const {
        return role == other.role && content == other.content &&
               content_parts == other.content_parts && tool_calls == other.tool_calls &&
               reasoning_content == other.reasoning_content &&
               tool_name == other.tool_name && tool_call_id == other.tool_call_id;
    }
    bool operator!=(const ChatMsg& other) const {
        return !(*this == other);
    }
    
    /**
     * Convert to OpenAI-compatible JSON format
     */
    json toJson() const;
    
    /**
     * Parse from OpenAI-compatible JSON format
     */
    static ChatMsg fromJson(const json& j);
};

// ============================================================================
// Streaming Diff Computation
// ============================================================================

/**
 * Represents a delta/diff between two message states
 * Used for streaming SSE responses
 */
struct ChatMsgDiff {
    std::string reasoning_content_delta;
    std::string content_delta;
    size_t tool_call_index = std::string::npos;  // npos means no tool call update
    ToolCall tool_call_delta;  // Partial tool call update
    
    bool empty() const {
        return reasoning_content_delta.empty() && 
               content_delta.empty() && 
               tool_call_index == std::string::npos;
    }
    
    bool operator==(const ChatMsgDiff& other) const {
        return reasoning_content_delta == other.reasoning_content_delta &&
               content_delta == other.content_delta &&
               tool_call_index == other.tool_call_index &&
               tool_call_delta == other.tool_call_delta;
    }
    
    /**
     * Compute diffs between previous and new message states
     * @param msg_prev Previous message state
     * @param msg_new New message state
     * @return List of diffs (may be multiple for tool calls)
     */
    static std::vector<ChatMsgDiff> computeDiffs(const ChatMsg& msg_prev, const ChatMsg& msg_new);
    
    /**
     * Convert diff to OpenAI streaming delta format
     */
    json toJsonDelta() const;
};

/**
 * Result of parsing streaming output for tool calls
 */
struct StreamingParseResult {
    std::string content;             // Non-tool-call content (after thinking)
    std::string reasoning_content;   // Thinking/reasoning content (DeepSeek R1, etc.)
    std::vector<ToolCall> tool_calls;
    bool has_tool_calls = false;
    bool has_reasoning = false;      // Whether reasoning content was detected
    bool parsing_complete = false;
    bool thinking_forced_open = false;  // Thinking tag opened but not closed
    
    /**
     * Convert to ChatMsg for diff computation
     */
    ChatMsg toChatMsg() const;
};

/**
 * Chat template capabilities detected from the template
 */
struct ChatTemplateCaps {
    bool supports_tools = false;
    bool supports_tool_calls = false;
    bool supports_tool_responses = false;
    bool supports_system_role = false;
    bool supports_parallel_tool_calls = false;
    bool requires_object_arguments = false;
    bool supports_reasoning = false;  // Whether template supports thinking/reasoning
};

/**
 * ChatHandler - Unified handler for chat templates and tool calls
 * 
 * This class encapsulates:
 * - Jinja2 chat template rendering using minja library
 * - Tool call parsing from model output (streaming and non-streaming)
 * - OpenAI-compatible message format handling
 */
class ChatHandler {
public:
    /**
     * Construct a ChatHandler with a Jinja2 template
     * @param template_source The Jinja2 template source code
     * @param bos_token Beginning of sequence token
     * @param eos_token End of sequence token
     */
    ChatHandler(const std::string& template_source, 
                const std::string& bos_token = "",
                const std::string& eos_token = "");
    
    ~ChatHandler();

    /**
     * Check if the handler has a valid template
     */
    bool hasTemplate() const;

    /**
     * Get the detected capabilities of the chat template
     */
    ChatTemplateCaps getCapabilities() const;

    /**
     * Apply the chat template to messages
     * @param messages OpenAI-format messages array
     * @param tools Optional tools array
     * @param add_generation_prompt Whether to add generation prompt
     * @param extra_context Additional context variables
     * @return Rendered prompt string
     */
    std::string applyTemplate(
        const json& messages,
        const json& tools = json(),
        bool add_generation_prompt = true,
        const json& extra_context = json()) const;

    /**
     * Parse tool calls from complete model output
     * @param output The complete model output string
     * @return Parsed tool calls
     */
    std::vector<ToolCall> parseToolCalls(const std::string& output) const;

    /**
     * Create a streaming tool call parser
     * @return Parser function that accepts chunks and returns parse results
     */
    std::function<StreamingParseResult(const std::string&)> createStreamingParser() const;

    /**
     * Convert tool calls to OpenAI format JSON
     * @param tool_calls Parsed tool calls
     * @return JSON array in OpenAI tool_calls format
     */
    static json toolCallsToJson(const std::vector<ToolCall>& tool_calls);

    /**
     * Build assistant message JSON with optional reasoning content
     * @param content The main content
     * @param reasoning_content Optional reasoning/thinking content
     * @param tool_calls Optional tool calls
     * @return JSON object in OpenAI message format
     */
    static json buildAssistantMessage(
        const std::string& content,
        const std::string& reasoning_content = "",
        const std::vector<ToolCall>& tool_calls = {});

    /**
     * Generate a unique tool call ID
     * @param index Optional index for the tool call
     * @return Unique ID string
     */
    static std::string generateToolCallId(int index = -1);

    /**
     * Get template source code
     */
    std::string getTemplateSource() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// PEG Parser for Structured Output Parsing
// ============================================================================

/**
 * PEG Parser Result - captures named groups from parsing
 */
struct PegParseResult {
    bool success = false;
    std::string matched;              // Matched text
    size_t start_pos = 0;             // Start position in input
    size_t end_pos = 0;               // End position in input
    std::string error_message;
    
    // Named captures for custom patterns
    std::map<std::string, std::string> captures;
};

/**
 * PEG Parser Rule - building block for patterns
 */
class PegRule {
public:
    virtual ~PegRule() = default;
    virtual std::string getName() const = 0;
};

/**
 * PegParser - Lightweight PEG parser for tool call extraction
 * 
 * Supports:
 * - Literal matching (case-sensitive and insensitive)
 * - Until delimiter matching
 * - Sequences and alternatives (choice)
 * - Repetition (zeroOrMore, oneOrMore, repeat with min/max)
 * - Optional matching
 * - Named captures
 * - Regular expressions
 * 
 * Example usage:
 *   PegParser parser;
 *   auto tool_call = parser.sequence({
 *       parser.literal("<tool_call>"),
 *       parser.capture("json", parser.until("</tool_call>", false)),
 *       parser.literal("</tool_call>")
 *   });
 *   parser.setRoot(tool_call);
 *   auto result = parser.parse(input);
 */
class PegParser {
public:
    PegParser();
    ~PegParser();
    
    // === Rule Builders (Static Factory Methods) ===
    
    /** Match exact literal string */
    static std::shared_ptr<PegRule> literal(const std::string& text);
    
    /** Match literal string ignoring case */
    static std::shared_ptr<PegRule> literalIgnoreCase(const std::string& text);
    
    /** Match until delimiter (include_delimiter: whether to consume delimiter) */
    static std::shared_ptr<PegRule> until(const std::string& delimiter, bool include_delimiter = false);
    
    /** Match regex pattern */
    static std::shared_ptr<PegRule> regex(const std::string& pattern);
    
    /** Match sequence of rules in order */
    static std::shared_ptr<PegRule> sequence(std::initializer_list<std::shared_ptr<PegRule>> rules);
    
    /** Match one of multiple alternatives (first match wins) */
    static std::shared_ptr<PegRule> choice(std::initializer_list<std::shared_ptr<PegRule>> rules);
    
    /** Match rule zero or one time */
    static std::shared_ptr<PegRule> optional(std::shared_ptr<PegRule> rule);
    
    /** Match rule zero or more times */
    static std::shared_ptr<PegRule> zeroOrMore(std::shared_ptr<PegRule> rule);
    
    /** Match rule one or more times */
    static std::shared_ptr<PegRule> oneOrMore(std::shared_ptr<PegRule> rule);
    
    /** Match rule between min and max times */
    static std::shared_ptr<PegRule> repeat(std::shared_ptr<PegRule> rule, size_t min_count, size_t max_count);
    
    /** Capture matched content with a name */
    static std::shared_ptr<PegRule> capture(const std::string& name, std::shared_ptr<PegRule> rule);
    
    /** Match any single character (or count characters) */
    static std::shared_ptr<PegRule> anyChar(size_t count = 1);
    
    // === Parser Configuration ===
    
    /** Set the root rule for parsing */
    void setRoot(std::shared_ptr<PegRule> rule);
    
    // === Parsing ===
    
    /** Parse input starting at position, return first match */
    PegParseResult parse(const std::string& input, size_t start_pos = 0) const;
    
    /** Find all matches in input */
    std::vector<PegParseResult> parseAll(const std::string& input) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Pre-built PEG parsers for common formats
 */
namespace peg_parsers {
    /** Create parser for Qwen3 format: <tool_call>JSON</tool_call> */
    std::unique_ptr<PegParser> createQwen3Parser();
    
    /** Create parser for DeepSeek format: <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜> */
    std::unique_ptr<PegParser> createDeepSeekParser();
    
    /** Create parser for thinking tags: <think>...</think> or <thinking>...</thinking> */
    std::unique_ptr<PegParser> createThinkingParser();
    
    /** Create parser for JSON code blocks: ```json ... ``` */
    std::unique_ptr<PegParser> createJsonBlockParser();
}

/**
 * StreamingToolCallParser - Stateful parser for streaming tool call detection
 * 
 * Supports multiple formats:
 * - Standard JSON format: {"name": "...", "arguments": {...}}
 * - Qwen3 format: <tool_call>...</tool_call>
 * - DeepSeek format: <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>
 * 
 * Also supports reasoning/thinking extraction:
 * - <think>...</think> (DeepSeek R1, Hermes, etc.)
 * - <thinking>...</thinking>
 * - <｜thinking｜>...<｜/thinking｜> (DeepSeek alternative)
 */
class StreamingToolCallParser {
public:
    StreamingToolCallParser();
    ~StreamingToolCallParser();

    /**
     * Feed a chunk of text to the parser
     * @param chunk New text chunk from model output
     * @return Parse result with any completed tool calls
     */
    StreamingParseResult feed(const std::string& chunk);

    /**
     * Finalize parsing (call when stream ends)
     * @return Final parse result
     */
    StreamingParseResult finalize();

    /**
     * Reset parser state for reuse
     */
    void reset();

    /**
     * Check if currently parsing a tool call
     */
    bool isParsingToolCall() const;
    
    /**
     * Get current accumulated message (for diff computation)
     */
    ChatMsg getCurrentMessage() const;
    
    /**
     * Compute diff from last message state
     * @param last_msg Previous message state to compare against
     * @return Delta since last state
     */
    std::vector<ChatMsgDiff> computeDiffSinceLastCall(const ChatMsg& last_msg) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Streaming Diff Helper
// ============================================================================

/**
 * StreamingDiffTracker - Tracks message state for incremental SSE updates
 */
class StreamingDiffTracker {
public:
    StreamingDiffTracker();
    ~StreamingDiffTracker();
    
    /**
     * Update with new message state
     * @param msg New message state
     * @return List of diffs to send
     */
    std::vector<ChatMsgDiff> update(const ChatMsg& msg);
    
    /**
     * Update with parse result
     * @param result Parse result from streaming parser
     * @return List of diffs to send
     */
    std::vector<ChatMsgDiff> update(const StreamingParseResult& result);
    
    /**
     * Reset tracker state
     */
    void reset();
    
    /**
     * Get current message state
     */
    const ChatMsg& getCurrentMessage() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fastllm

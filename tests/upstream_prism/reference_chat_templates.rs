//! Clean-room independent safe Rust chat template inventory and renderer.
//!
//! Implements upstream `llama_chat_builtin_templates`, `llama_chat_apply_template`
//! legacy template detection/rendering, and `common_chat_templates_apply` Jinja
//! rendering (with capability probing and string/typed content fallback) using
//! `reference_jinja`, without importing `bonsai_prism_runtime`, invoking a native
//! bridge, or reading expected-answer fixtures.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use super::reference_jinja::{
    self, JVal, Program, eval_program_with_now, execute_caps_flags, parse_template_with_limit,
};
use super::{Result, invalid};

/// Upstream source identifier covered by this independent reference provider.
pub(super) const ID: &str = "tests/test-chat-template.cpp";
const SOURCE_SHA: &str = "7a790605222858ca5d5a2de94de27115d693e2c293638c392b7726a44ad3451f";

const MAX_OPERATIONS: usize = 128;
const MAX_BUILTIN_CAPACITY: usize = 4096;
const MAX_LEGACY_CAPACITY: usize = 131_072;
const MAX_TEMPLATE_BYTES: usize = 256 * 1024;
const MAX_TOKEN_BYTES: usize = 4096;
const MAX_MESSAGES: usize = 1024;
const MAX_ROLE_BYTES: usize = 4096;
const MAX_CONTENT_BYTES: usize = 65_536;
const MAX_MESSAGES_TOTAL_BYTES: usize = 1024 * 1024;

const CHATML_JINJA_TEMPLATE: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

const BUILTIN_TEMPLATES: &[&str] = &[
    "bailing",
    "bailing-think",
    "chatglm3",
    "chatglm4",
    "chatml",
    "command-r",
    "deepseek",
    "deepseek2",
    "deepseek3",
    "exaone-moe",
    "exaone3",
    "exaone4",
    "falcon3",
    "gemma",
    "gigachat",
    "glmedge",
    "granite",
    "grok-2",
    "hunyuan-dense",
    "hunyuan-moe",
    "kimi-k2",
    "llama2",
    "llama2-sys",
    "llama2-sys-bos",
    "llama2-sys-strip",
    "llama3",
    "megrez",
    "minicpm",
    "mistral-v1",
    "mistral-v3",
    "mistral-v3-tekken",
    "mistral-v7",
    "mistral-v7-tekken",
    "monarch",
    "openai-moe",
    "openchat",
    "orion",
    "pangu-embedded",
    "phi3",
    "phi4",
    "rwkv-world",
    "seed_oss",
    "solar-open",
    "vicuna",
    "vicuna-orca",
    "yandex",
    "zephyr",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyFormat {
    Chatml,
    Llama2,
    Llama2Sys,
    Llama2SysBos,
    Llama2SysStrip,
    MistralV1,
    MistralV3,
    MistralV3Tekken,
    MistralV7,
    MistralV7Tekken,
    Phi3,
    Phi4,
    Falcon3,
    Zephyr,
    Monarch,
    Gemma,
    Orion,
    Openchat,
    Vicuna,
    VicunaOrca,
    Deepseek,
    Deepseek2,
    Deepseek3,
    CommandR,
    Llama3,
    Chatglm3,
    Chatglm4,
    GlmEdge,
    Minicpm,
    Exaone3,
    Exaone4,
    ExaoneMoe,
    RwkvWorld,
    Granite,
    Gigachat,
    Megrez,
    Yandex,
    Bailing,
    BailingThink,
    SeedOss,
    Grok2,
    HunyuanMoe,
    HunyuanDense,
    OpenaiMoe,
    KimiK2,
    PanguEmbedded,
    SolarOpen,
}

/// Borrowed chat message used by clean-room template renderers.
#[derive(Clone, Debug)]
pub(super) struct ChatMessage<'a> {
    /// Message role (`"system"`, `"user"`, `"assistant"`, `"tool"`, etc.).
    pub(super) role: &'a str,
    /// Message text content.
    pub(super) content: &'a str,
}

/// Returns the sorted list of upstream built-in legacy template names.
pub(super) const fn builtin_template_names() -> &'static [&'static str] {
    BUILTIN_TEMPLATES
}

fn format_from_name(name: &str) -> Option<LegacyFormat> {
    match name {
        "chatml" => Some(LegacyFormat::Chatml),
        "llama2" => Some(LegacyFormat::Llama2),
        "llama2-sys" => Some(LegacyFormat::Llama2Sys),
        "llama2-sys-bos" => Some(LegacyFormat::Llama2SysBos),
        "llama2-sys-strip" => Some(LegacyFormat::Llama2SysStrip),
        "mistral-v1" => Some(LegacyFormat::MistralV1),
        "mistral-v3" => Some(LegacyFormat::MistralV3),
        "mistral-v3-tekken" => Some(LegacyFormat::MistralV3Tekken),
        "mistral-v7" => Some(LegacyFormat::MistralV7),
        "mistral-v7-tekken" => Some(LegacyFormat::MistralV7Tekken),
        "phi3" => Some(LegacyFormat::Phi3),
        "phi4" => Some(LegacyFormat::Phi4),
        "falcon3" => Some(LegacyFormat::Falcon3),
        "zephyr" => Some(LegacyFormat::Zephyr),
        "monarch" => Some(LegacyFormat::Monarch),
        "gemma" => Some(LegacyFormat::Gemma),
        "orion" => Some(LegacyFormat::Orion),
        "openchat" => Some(LegacyFormat::Openchat),
        "vicuna" => Some(LegacyFormat::Vicuna),
        "vicuna-orca" => Some(LegacyFormat::VicunaOrca),
        "deepseek" => Some(LegacyFormat::Deepseek),
        "deepseek2" => Some(LegacyFormat::Deepseek2),
        "deepseek3" => Some(LegacyFormat::Deepseek3),
        "command-r" => Some(LegacyFormat::CommandR),
        "llama3" => Some(LegacyFormat::Llama3),
        "chatglm3" => Some(LegacyFormat::Chatglm3),
        "chatglm4" => Some(LegacyFormat::Chatglm4),
        "glmedge" => Some(LegacyFormat::GlmEdge),
        "minicpm" => Some(LegacyFormat::Minicpm),
        "exaone3" => Some(LegacyFormat::Exaone3),
        "exaone4" => Some(LegacyFormat::Exaone4),
        "exaone-moe" => Some(LegacyFormat::ExaoneMoe),
        "rwkv-world" => Some(LegacyFormat::RwkvWorld),
        "granite" => Some(LegacyFormat::Granite),
        "gigachat" => Some(LegacyFormat::Gigachat),
        "megrez" => Some(LegacyFormat::Megrez),
        "yandex" => Some(LegacyFormat::Yandex),
        "bailing" => Some(LegacyFormat::Bailing),
        "bailing-think" => Some(LegacyFormat::BailingThink),
        "seed_oss" => Some(LegacyFormat::SeedOss),
        "grok-2" => Some(LegacyFormat::Grok2),
        "hunyuan-moe" => Some(LegacyFormat::HunyuanMoe),
        "hunyuan-dense" => Some(LegacyFormat::HunyuanDense),
        "openai-moe" => Some(LegacyFormat::OpenaiMoe),
        "kimi-k2" => Some(LegacyFormat::KimiK2),
        "pangu-embedded" => Some(LegacyFormat::PanguEmbedded),
        "solar-open" => Some(LegacyFormat::SolarOpen),
        _ => None,
    }
}

fn detect_legacy_other(tmpl: &str) -> Option<LegacyFormat> {
    let has = |needle: &str| tmpl.contains(needle);
    if has("<用户>") {
        return Some(LegacyFormat::Minicpm);
    }
    if has("<seed:bos>") {
        return Some(LegacyFormat::SeedOss);
    }
    if has("<|separator|>") {
        return Some(LegacyFormat::Grok2);
    }
    if has("<|extra_4|>") && has("<|extra_0|>") {
        return Some(LegacyFormat::HunyuanMoe);
    }
    if has("<|startoftext|>") && has("<|extra_0|>") {
        return Some(LegacyFormat::HunyuanDense);
    }
    if has("<|start|>") && has("<|message|>") {
        return Some(LegacyFormat::OpenaiMoe);
    }
    if has("<role>") && has("</role>") {
        return Some(if has("<thought>") || has("<think>") {
            LegacyFormat::BailingThink
        } else {
            LegacyFormat::Bailing
        });
    }
    if has("<|role_sep|>") || (has("additional_special_tokens") && has("available functions")) {
        return Some(LegacyFormat::Gigachat);
    }
    if has("[SEP]") && has("Ассистент") {
        return Some(LegacyFormat::Yandex);
    }
    if has("[unused9]") || has("[unused10]") {
        return Some(LegacyFormat::PanguEmbedded);
    }
    if has("<|tm_start|>") || has("<|im_middle|>") {
        return Some(LegacyFormat::KimiK2);
    }
    if has("Human:") && has("Assistant:") {
        return Some(LegacyFormat::Orion);
    }
    if has("USER:") && has("ASSISTANT:") {
        return Some(if has("SYSTEM:") {
            LegacyFormat::VicunaOrca
        } else {
            LegacyFormat::Vicuna
        });
    }
    if has("User:") && has("Assistant:") {
        return Some(LegacyFormat::RwkvWorld);
    }
    if has("eos_token") && has("assistant\\n") {
        return Some(LegacyFormat::Monarch);
    }
    detect_legacy_inst(tmpl)
}

fn detect_legacy_inst(tmpl: &str) -> Option<LegacyFormat> {
    let has = |needle: &str| tmpl.contains(needle);
    if !has("[INST]") {
        return None;
    }
    if has("[SYSTEM_PROMPT]") {
        return Some(if has("[INST] ") {
            LegacyFormat::MistralV7
        } else {
            LegacyFormat::MistralV7Tekken
        });
    }
    if has("<<SYS>>") {
        if has(".strip()") {
            return Some(LegacyFormat::Llama2SysStrip);
        }
        if has("idx > 1") {
            return Some(LegacyFormat::Llama2Sys);
        }
        if has("bos_token") {
            return Some(LegacyFormat::Llama2SysBos);
        }
        return Some(LegacyFormat::Llama2Sys);
    }
    if has(" [INST] ") {
        return Some(LegacyFormat::MistralV1);
    }
    if has("[INST] ") && has(" [/INST]") {
        return Some(LegacyFormat::Llama2);
    }
    if has("[INST] ") {
        return Some(LegacyFormat::MistralV3);
    }
    Some(LegacyFormat::MistralV3Tekken)
}

fn detect_legacy_template(tmpl: &str) -> Option<LegacyFormat> {
    if let Some(exact) = format_from_name(tmpl) {
        return Some(exact);
    }
    let has = |needle: &str| tmpl.contains(needle);
    if has("<|im_start|>") {
        return Some(if has("<|im_sep|>") {
            LegacyFormat::Phi4
        } else {
            LegacyFormat::Chatml
        });
    }
    if has("<|start_header_id|>") {
        return Some(LegacyFormat::Llama3);
    }
    if has("[|endofturn|]") {
        if has("<thought>") {
            return Some(LegacyFormat::Exaone4);
        }
        if has("<|thought|>") || has("exaone-moe") {
            return Some(LegacyFormat::ExaoneMoe);
        }
        return Some(LegacyFormat::Exaone3);
    }
    if has("<|START_OF_TURN_TOKEN|>") {
        return Some(LegacyFormat::CommandR);
    }
    if has("<start_of_turn>") {
        return Some(LegacyFormat::Gemma);
    }
    if has("<|start_of_role|>") {
        return Some(LegacyFormat::Granite);
    }
    if has("<|role_start|>") {
        return Some(LegacyFormat::Megrez);
    }
    if has("<|end|>") && (has("<|user|>") || has("<|assistant|>")) {
        return Some(LegacyFormat::Phi3);
    }
    if has("[gMASK]<sop>") {
        return Some(LegacyFormat::Chatglm4);
    }
    if has("[gMASK]sop") {
        return Some(LegacyFormat::Chatglm3);
    }
    if has("<|system|>") && has("<|user|>") && has("<|assistant|>") {
        return Some(LegacyFormat::GlmEdge);
    }
    if has("<|user|>") && has("<|assistant|>") {
        if has("<|observation|>") {
            return Some(LegacyFormat::GlmEdge);
        }
        if has("</s>") {
            return Some(LegacyFormat::Zephyr);
        }
        return Some(LegacyFormat::Falcon3);
    }
    if has("<｜User｜>") {
        return Some(LegacyFormat::Deepseek3);
    }
    if has("Assistant:") && has("User:") && (has("<｜end▁of▁sentence｜>") || has("eos_token"))
    {
        return Some(LegacyFormat::Deepseek2);
    }
    if has("### Instruction:") && has("<|EOT|>") {
        return Some(LegacyFormat::Deepseek);
    }
    if has("GPT4 Correct ") {
        return Some(LegacyFormat::Openchat);
    }
    detect_legacy_other(tmpl)
}

fn trim_ascii_whitespace(text: &str) -> &str {
    text.trim()
}

fn render_llama2_family(fmt: LegacyFormat, messages: &[ChatMessage<'_>], add_ass: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let mut system_prompt = String::new();
    let mut add_sys = false;
    let is_sys = matches!(
        fmt,
        LegacyFormat::Llama2Sys | LegacyFormat::Llama2SysBos | LegacyFormat::Llama2SysStrip
    );
    let add_bos = matches!(fmt, LegacyFormat::Llama2SysBos);
    let no_turn_bos = matches!(fmt, LegacyFormat::Llama2 | LegacyFormat::Llama2SysStrip);
    let trailing_inst_space = false;
    let no_eos_space = true;
    let no_ass_leading_space = true;
    let inside_strip = matches!(fmt, LegacyFormat::Llama2SysStrip);
    let mut is_first_user = true;

    for (idx, msg) in messages.iter().enumerate() {
        let content = if inside_strip {
            trim_ascii_whitespace(msg.content)
        } else {
            msg.content
        };
        if idx == 0 && msg.role == "system" {
            if is_sys {
                system_prompt = format!("<<SYS>>\n{content}\n<</SYS>>\n\n");
            } else {
                system_prompt = format!("{content}\n");
            }
            add_sys = true;
            continue;
        }
        if msg.role == "user" {
            if (is_first_user && add_bos) || (!is_first_user && !no_turn_bos) {
                out.push_str("<s>");
            }
            is_first_user = false;
            out.push_str("[INST] ");
            if add_sys {
                out.push_str(&system_prompt);
                add_sys = false;
            }
            if trailing_inst_space {
                let _ = write!(out, "{content} [/INST] ");
            } else {
                let _ = write!(out, "{content} [/INST]");
            }
        } else if msg.role == "assistant" {
            let lead = if no_ass_leading_space { "" } else { " " };
            let tail = if no_eos_space { "</s>" } else { " </s>" };
            let _ = write!(out, "{lead}{content}{tail}");
        } else {
            let _ = write!(out, "{content}");
        }
    }
    if add_ass && messages.last().is_some_and(|m| m.role == "system") && add_sys {
        if add_bos {
            out.push_str("<s>");
        }
        let _ = write!(out, "[INST] {system_prompt} [/INST]");
    }
    out
}

fn render_mistral_family(fmt: LegacyFormat, messages: &[ChatMessage<'_>], add_ass: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let mut system_prompt = String::new();
    let leading_space = matches!(
        fmt,
        LegacyFormat::MistralV1 | LegacyFormat::MistralV3 | LegacyFormat::MistralV7
    );
    let trailing_space = matches!(fmt, LegacyFormat::MistralV1);
    let is_v7 = matches!(fmt, LegacyFormat::MistralV7 | LegacyFormat::MistralV7Tekken);
    let inst_open = match fmt {
        LegacyFormat::MistralV1 => " [INST] ",
        LegacyFormat::MistralV3 | LegacyFormat::MistralV7 => "[INST] ",
        _ => "[INST]",
    };
    let inst_close = if trailing_space {
        " [/INST]"
    } else {
        "[/INST]"
    };

    for msg in messages {
        if msg.role == "system" {
            if is_v7 {
                if matches!(fmt, LegacyFormat::MistralV7) {
                    let _ = write!(out, "[SYSTEM_PROMPT] {}[/SYSTEM_PROMPT]", msg.content);
                } else {
                    let _ = write!(out, "[SYSTEM_PROMPT]{}[/SYSTEM_PROMPT]", msg.content);
                }
            } else {
                if !system_prompt.is_empty() {
                    system_prompt.push_str("\n\n");
                }
                system_prompt.push_str(msg.content);
            }
        } else if msg.role == "user" {
            if system_prompt.is_empty() {
                let _ = write!(out, "{inst_open}{}{inst_close}", msg.content);
            } else {
                let _ = write!(
                    out,
                    "{inst_open}{system_prompt}\n\n{}{inst_close}",
                    msg.content
                );
                system_prompt.clear();
            }
        } else if msg.role == "assistant" {
            let content = if matches!(fmt, LegacyFormat::MistralV3) {
                msg.content.trim()
            } else {
                msg.content
            };
            if leading_space {
                let _ = write!(out, " {content}</s>");
            } else {
                let _ = write!(out, "{content}</s>");
            }
        } else {
            let _ = write!(out, "{}", msg.content);
        }
    }
    let _ = add_ass;
    out
}

fn render_basic_tagged_legacy(
    fmt: LegacyFormat,
    messages: &[ChatMessage<'_>],
    add_ass: bool,
    out: &mut String,
) {
    use std::fmt::Write as _;

    match fmt {
        LegacyFormat::Chatml => {
            for msg in messages {
                let _ = writeln!(out, "<|im_start|>{}\n{}<|im_end|>", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("<|im_start|>assistant\n");
            }
        }
        LegacyFormat::Phi3 => {
            for msg in messages {
                let _ = writeln!(out, "<|{}|>\n{}<|end|>", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("<|assistant|>\n");
            }
        }
        LegacyFormat::Phi4 => {
            for msg in messages {
                let _ = write!(
                    out,
                    "<|im_start|>{}<|im_sep|>{}<|im_end|>",
                    msg.role, msg.content
                );
            }
            if add_ass {
                out.push_str("<|im_start|>assistant<|im_sep|>");
            }
        }
        LegacyFormat::Falcon3 | LegacyFormat::PanguEmbedded | LegacyFormat::SolarOpen => {
            for msg in messages {
                let _ = writeln!(out, "<|{}|>\n{}", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("<|assistant|>\n");
            }
        }
        LegacyFormat::Zephyr => {
            for msg in messages {
                let _ = writeln!(out, "<|{}|>\n{}</s>", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("<|assistant|>\n");
            }
        }
        LegacyFormat::Monarch => {
            for (idx, msg) in messages.iter().enumerate() {
                if idx > 0 {
                    out.push_str("<s>");
                }
                let _ = writeln!(out, "{}\n{}</s>", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("<s>assistant\n");
            }
        }
        LegacyFormat::Gemma => {
            let mut system_prompt = String::new();
            for msg in messages {
                if msg.role == "system" {
                    system_prompt = msg.content.trim().to_owned();
                    continue;
                }
                let role = if msg.role == "assistant" {
                    "model"
                } else {
                    msg.role
                };
                out.push_str("<start_of_turn>");
                out.push_str(role);
                out.push('\n');
                if !system_prompt.is_empty() && msg.role != "assistant" {
                    out.push_str(&system_prompt);
                    out.push_str("\n\n");
                    system_prompt.clear();
                }
                out.push_str(msg.content.trim());
                out.push_str("<end_of_turn>\n");
            }
            if add_ass {
                out.push_str("<start_of_turn>model\n");
            }
        }
        _ => {}
    }
}

fn render_role_colon_legacy(
    fmt: LegacyFormat,
    messages: &[ChatMessage<'_>],
    add_ass: bool,
    out: &mut String,
) {
    use std::fmt::Write as _;

    match fmt {
        LegacyFormat::Orion => {
            let mut system_prompt = String::new();
            for msg in messages {
                if msg.role == "system" {
                    system_prompt = format!("{}\n\n", msg.content);
                } else if msg.role == "user" {
                    let _ = write!(
                        out,
                        "Human: {system_prompt}{}\n\nAssistant: </s>",
                        msg.content
                    );
                    system_prompt.clear();
                } else if msg.role == "assistant" {
                    let _ = write!(out, "{}</s>", msg.content);
                }
            }
        }
        LegacyFormat::Openchat => {
            for msg in messages {
                if msg.role == "system" {
                    let _ = write!(out, "{}<|end_of_turn|>", msg.content);
                } else {
                    let role = match msg.role {
                        "user" => "User",
                        "assistant" => "Assistant",
                        other => other,
                    };
                    let _ = write!(out, "GPT4 Correct {role}: {}<|end_of_turn|>", msg.content);
                }
            }
            if add_ass {
                out.push_str("GPT4 Correct Assistant:");
            }
        }
        LegacyFormat::Vicuna | LegacyFormat::VicunaOrca => {
            for msg in messages {
                if msg.role == "system" {
                    if matches!(fmt, LegacyFormat::VicunaOrca) {
                        let _ = writeln!(out, "SYSTEM: {}", msg.content);
                    } else {
                        let _ = write!(out, "{}\n\n", msg.content);
                    }
                } else if msg.role == "user" {
                    let _ = writeln!(out, "USER: {}", msg.content);
                } else if msg.role == "assistant" {
                    let _ = writeln!(out, "ASSISTANT: {}</s>", msg.content);
                }
            }
            if add_ass {
                out.push_str("ASSISTANT:");
            }
        }
        LegacyFormat::Deepseek => {
            for msg in messages {
                if msg.role == "system" {
                    out.push_str(msg.content);
                } else if msg.role == "user" {
                    let _ = writeln!(out, "### Instruction:\n{}", msg.content);
                } else if msg.role == "assistant" {
                    let _ = writeln!(out, "### Response:\n{}\n<|EOT|>", msg.content);
                }
            }
            if add_ass {
                out.push_str("### Response:\n");
            }
        }
        LegacyFormat::Deepseek2 => {
            for msg in messages {
                if msg.role == "system" {
                    let _ = write!(out, "{}\n\n", msg.content);
                } else if msg.role == "user" {
                    let _ = write!(out, "User: {}\n\n", msg.content);
                } else if msg.role == "assistant" {
                    let _ = write!(out, "Assistant: {}<｜end▁of▁sentence｜>", msg.content);
                }
            }
            if add_ass {
                out.push_str("Assistant:");
            }
        }
        LegacyFormat::Deepseek3 => {
            for msg in messages {
                if msg.role == "system" {
                    out.push_str(msg.content);
                } else if msg.role == "user" {
                    let _ = write!(out, "<｜User｜>{}", msg.content);
                } else if msg.role == "assistant" {
                    let _ = write!(out, "<｜Assistant｜>{}<｜end▁of▁sentence｜>", msg.content);
                }
            }
            if add_ass {
                out.push_str("<｜Assistant｜>");
            }
        }
        _ => {}
    }
}

fn render_command_glm_exaone_legacy(
    fmt: LegacyFormat,
    messages: &[ChatMessage<'_>],
    add_ass: bool,
    out: &mut String,
) {
    use std::fmt::Write as _;

    match fmt {
        LegacyFormat::CommandR => {
            for msg in messages {
                let role_tok = match msg.role {
                    "system" => "SYSTEM_TOKEN",
                    "user" => "USER_TOKEN",
                    "assistant" => "CHATBOT_TOKEN",
                    other => other,
                };
                let content = if msg.role == "system" {
                    msg.content
                } else {
                    msg.content.trim()
                };
                let _ = write!(
                    out,
                    "<|START_OF_TURN_TOKEN|><|{role_tok}|>{content}<|END_OF_TURN_TOKEN|>"
                );
            }
            if add_ass {
                out.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
            }
        }
        LegacyFormat::Llama3 => {
            for msg in messages {
                let _ = write!(
                    out,
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    msg.role,
                    msg.content.trim()
                );
            }
            if add_ass {
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        }
        LegacyFormat::Chatglm3 => {
            out.push_str("[gMASK]sop");
            for msg in messages {
                let _ = write!(out, "<|{}|>\n {}", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("<|assistant|>");
            }
        }
        LegacyFormat::Chatglm4 => {
            out.push_str("[gMASK]<sop>");
            for msg in messages {
                let _ = write!(out, "<|{}|>\n{}", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("<|assistant|>\n");
            }
        }
        LegacyFormat::GlmEdge => {
            for msg in messages {
                let _ = write!(out, "<|{}|>\n{}", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("<|assistant|>");
            }
        }
        LegacyFormat::Minicpm => {
            for msg in messages {
                let content = msg.content.trim();
                if msg.role == "user" {
                    let _ = write!(out, "<用户>{content}<AI>");
                } else {
                    out.push_str(content);
                }
            }
        }
        LegacyFormat::Exaone3 | LegacyFormat::Exaone4 | LegacyFormat::ExaoneMoe => {
            for msg in messages {
                let _ = writeln!(out, "[|{}|]{}[|endofturn|]", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("[|assistant|]");
                if matches!(fmt, LegacyFormat::Exaone4) {
                    out.push_str("<thought>\n\n</thought>\n");
                }
            }
        }
        _ => {}
    }
}

fn render_rwkv_granite_giga_megrez_yandex(
    fmt: LegacyFormat,
    messages: &[ChatMessage<'_>],
    add_ass: bool,
    out: &mut String,
) {
    use std::fmt::Write as _;

    match fmt {
        LegacyFormat::RwkvWorld => {
            for msg in messages {
                let role = if msg.role == "user" {
                    "User"
                } else if msg.role == "assistant" {
                    "Assistant"
                } else {
                    msg.role
                };
                if msg.role == "system" {
                    let _ = write!(out, "{}\n\n", msg.content);
                } else {
                    let _ = write!(out, "{role}: {}\n\n", msg.content);
                }
            }
            if add_ass {
                out.push_str("Assistant:");
            }
        }
        LegacyFormat::Granite => {
            for msg in messages {
                if msg.role == "assistant_tool_call" {
                    if msg.content.contains("<tool_call>") {
                        let _ = writeln!(
                            out,
                            "<|start_of_role|>assistant<|end_of_role|><|tool_call|>{}<|end_of_text|>",
                            msg.content
                        );
                    } else {
                        let _ = writeln!(
                            out,
                            "<|start_of_role|>assistant_tool_call<|end_of_role|><|tool_call|>{}<|end_of_text|>",
                            msg.content
                        );
                    }
                } else {
                    let _ = writeln!(
                        out,
                        "<|start_of_role|>{}<|end_of_role|>{}<|end_of_text|>",
                        msg.role, msg.content
                    );
                }
            }
            if add_ass {
                out.push_str("<|start_of_role|>assistant<|end_of_role|>");
            }
        }
        LegacyFormat::Gigachat => {
            out.push_str("<s>");
            for (idx, msg) in messages.iter().enumerate() {
                if idx == 0 && msg.role == "system" {
                    let _ = write!(out, "{}<|message_sep|>", msg.content);
                } else if msg.role == "user" {
                    let _ = write!(
                        out,
                        "user<|role_sep|>{}<|message_sep|>available functions<|role_sep|>[]<|message_sep|>",
                        msg.content
                    );
                } else {
                    let _ = write!(
                        out,
                        "{}<|role_sep|>{}<|message_sep|>",
                        msg.role, msg.content
                    );
                }
            }
            if add_ass {
                out.push_str("assistant<|role_sep|>");
            }
        }
        LegacyFormat::Megrez => {
            for msg in messages {
                let _ = write!(
                    out,
                    "<|role_start|>{}<|role_end|>{}<|turn_end|>",
                    msg.role, msg.content
                );
            }
            if add_ass {
                out.push_str("<|role_start|>assistant<|role_end|>");
            }
        }
        LegacyFormat::Yandex => {
            for msg in messages {
                if msg.role == "user" {
                    let _ = write!(out, " Пользователь: {}\n\n", msg.content);
                } else if msg.role == "assistant" {
                    let _ = write!(out, " Ассистент: {}\n\n", msg.content);
                } else if msg.role != "system" {
                    let _ = write!(out, "{}: {}[SEP]", msg.role, msg.content);
                }
            }
            if add_ass {
                out.push_str(" Ассистент:[SEP]");
            }
        }
        _ => {}
    }
}

fn render_tail_legacy(
    fmt: LegacyFormat,
    messages: &[ChatMessage<'_>],
    add_ass: bool,
    out: &mut String,
) {
    use std::fmt::Write as _;

    match fmt {
        LegacyFormat::Bailing | LegacyFormat::BailingThink => {
            for msg in messages {
                let role = match msg.role {
                    "system" => "SYSTEM",
                    "user" => "HUMAN",
                    "assistant" => "ASSISTANT",
                    other => other,
                };
                let _ = write!(out, "<role>{role}</role>{}", msg.content);
            }
            if add_ass {
                out.push_str("<role>ASSISTANT</role>");
                if matches!(fmt, LegacyFormat::BailingThink) {
                    out.push_str("<think>\n");
                }
            }
        }
        LegacyFormat::SeedOss => {
            for msg in messages {
                if msg.role == "assistant" {
                    let trimmed = msg.content.trim();
                    let _ = write!(out, "<seed:bos>assistant");
                    if !trimmed.is_empty() {
                        let _ = write!(out, "\n{trimmed}<seed:eos>");
                    }
                } else {
                    let _ = write!(out, "<seed:bos>{}\n{}<seed:eos>", msg.role, msg.content);
                }
            }
            if add_ass {
                out.push_str("<seed:bos>assistant\n");
            }
        }
        LegacyFormat::Grok2 => {
            for msg in messages {
                let role = match msg.role {
                    "system" => "System",
                    "user" => "Human",
                    "assistant" => "Assistant",
                    other => other,
                };
                let _ = write!(out, "{role}: {}<|separator|>\n\n", msg.content);
            }
            if add_ass {
                out.push_str("Assistant:");
            }
        }
        LegacyFormat::HunyuanMoe | LegacyFormat::HunyuanDense => {
            for (idx, msg) in messages.iter().enumerate() {
                if idx == 0 && msg.role == "system" {
                    let _ = write!(out, "<|startoftext|>{}<|extra_4|>", msg.content);
                } else if msg.role == "user" {
                    let _ = write!(out, "{}<|extra_0|>", msg.content);
                } else if msg.role == "assistant" {
                    let _ = write!(out, "{}<|eos|>", msg.content);
                }
            }
        }
        LegacyFormat::OpenaiMoe => {
            for msg in messages {
                let end = if msg.role == "assistant" {
                    "<|return|>"
                } else {
                    "<|end|>"
                };
                let _ = write!(out, "<|start|>{}<|message|>{}{end}", msg.role, msg.content);
            }
            if add_ass {
                out.push_str("<|start|>assistant<|channel|>analysis<|message|>");
            }
        }
        LegacyFormat::KimiK2 => {
            for msg in messages {
                let _ = write!(
                    out,
                    "<|im_{}|>{}<|im_middle|>{}<|im_end|>",
                    msg.role, msg.role, msg.content
                );
            }
            if add_ass {
                out.push_str("<|im_assistant|>assistant<|im_middle|>");
            }
        }
        _ => {}
    }
}

/// Clean-room implementation of upstream `llama_chat_apply_template`.
///
/// Returns `None` if `template` is neither a known built-in template name nor
/// matched by the upstream substring heuristics.
pub(super) fn legacy_apply(
    template: &str,
    messages: &[ChatMessage<'_>],
    add_ass: bool,
) -> Option<String> {
    let fmt = detect_legacy_template(template)?;
    let mut out = String::new();
    match fmt {
        LegacyFormat::Llama2
        | LegacyFormat::Llama2Sys
        | LegacyFormat::Llama2SysBos
        | LegacyFormat::Llama2SysStrip => {
            out = render_llama2_family(fmt, messages, add_ass);
        }
        LegacyFormat::MistralV1
        | LegacyFormat::MistralV3
        | LegacyFormat::MistralV3Tekken
        | LegacyFormat::MistralV7
        | LegacyFormat::MistralV7Tekken => {
            out = render_mistral_family(fmt, messages, add_ass);
        }
        LegacyFormat::Chatml
        | LegacyFormat::Phi3
        | LegacyFormat::Phi4
        | LegacyFormat::Falcon3
        | LegacyFormat::PanguEmbedded
        | LegacyFormat::SolarOpen
        | LegacyFormat::Zephyr
        | LegacyFormat::Monarch
        | LegacyFormat::Gemma => {
            render_basic_tagged_legacy(fmt, messages, add_ass, &mut out);
        }
        LegacyFormat::Orion
        | LegacyFormat::Openchat
        | LegacyFormat::Vicuna
        | LegacyFormat::VicunaOrca
        | LegacyFormat::Deepseek
        | LegacyFormat::Deepseek2
        | LegacyFormat::Deepseek3 => {
            render_role_colon_legacy(fmt, messages, add_ass, &mut out);
        }
        LegacyFormat::CommandR
        | LegacyFormat::Llama3
        | LegacyFormat::Chatglm3
        | LegacyFormat::Chatglm4
        | LegacyFormat::GlmEdge
        | LegacyFormat::Minicpm
        | LegacyFormat::Exaone3
        | LegacyFormat::Exaone4
        | LegacyFormat::ExaoneMoe => {
            render_command_glm_exaone_legacy(fmt, messages, add_ass, &mut out);
        }
        LegacyFormat::RwkvWorld
        | LegacyFormat::Granite
        | LegacyFormat::Gigachat
        | LegacyFormat::Megrez
        | LegacyFormat::Yandex => {
            render_rwkv_granite_giga_megrez_yandex(fmt, messages, add_ass, &mut out);
        }
        LegacyFormat::Bailing
        | LegacyFormat::BailingThink
        | LegacyFormat::SeedOss
        | LegacyFormat::Grok2
        | LegacyFormat::HunyuanMoe
        | LegacyFormat::HunyuanDense
        | LegacyFormat::OpenaiMoe
        | LegacyFormat::KimiK2 => {
            render_tail_legacy(fmt, messages, add_ass, &mut out);
        }
    }
    Some(out)
}

/// Capabilities extracted from a parsed Jinja chat template.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Mirrors the six boolean capability flags of upstream common_chat_template_caps."
)]
pub(super) struct TemplateCaps {
    pub(super) supports_tools: bool,
    pub(super) supports_tool_calls: bool,
    pub(super) supports_system_role: bool,
    pub(super) supports_parallel_tool_calls: bool,
    pub(super) supports_typed_content: bool,
    pub(super) supports_string_content: bool,
}

fn build_message_jval(role: &str, content: &str, typed_content: bool) -> JVal {
    let content_val = if typed_content {
        if content.is_empty() {
            JVal::array(Vec::new())
        } else {
            JVal::array(vec![JVal::object(vec![
                (JVal::text_str("type"), JVal::bytes(b"text".to_vec(), true)),
                (
                    JVal::text_str("text"),
                    JVal::bytes(content.as_bytes().to_vec(), true),
                ),
            ])])
        }
    } else {
        JVal::bytes(content.as_bytes().to_vec(), true)
    };
    JVal::object(vec![
        (
            JVal::text_str("role"),
            JVal::bytes(role.as_bytes().to_vec(), true),
        ),
        (JVal::text_str("content"), content_val),
    ])
}

fn current_unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|dur| i64::try_from(dur.as_secs()).ok())
        .unwrap_or(0)
}

/// Probe template capabilities (`supports_*`) using clean-room Jinja execution.
pub(super) fn template_caps_from_program(program: &Program) -> TemplateCaps {
    let (supports_string_content, supports_typed_content) =
        execute_caps_flags(program, Some(0)).unwrap_or((false, false));
    let use_typed = !supports_string_content && supports_typed_content;

    let render_with = |msgs: JVal, tools: Option<JVal>| -> Option<Vec<u8>> {
        let mut pairs = vec![
            (JVal::text_str("messages"), msgs),
            (JVal::text_str("bos_token"), JVal::bytes(Vec::new(), true)),
            (JVal::text_str("eos_token"), JVal::bytes(Vec::new(), true)),
            (JVal::text_str("add_generation_prompt"), JVal::boolean(true)),
        ];
        if let Some(tools_val) = tools {
            pairs.push((JVal::text_str("tools"), tools_val));
        }
        let globals = JVal::object(pairs);
        let spans = eval_program_with_now(program, &globals, Some(0)).ok()?;
        Some(spans.into_iter().flat_map(|s| s.bytes).collect())
    };

    let sys_probe = render_with(
        JVal::array(vec![
            build_message_jval("system", "SYS_CAPS_MARKER", use_typed),
            build_message_jval("user", "USR_CAPS_MARKER", use_typed),
        ]),
        None,
    );
    let supports_system_role = sys_probe.is_some_and(|bytes| {
        bytes
            .windows(b"SYS_CAPS_MARKER".len())
            .any(|win| win == b"SYS_CAPS_MARKER")
    });

    let tools_json = r#"[{"type":"function","function":{"name":"caps_tool_alpha","description":"Alpha tool","parameters":{"type":"object","properties":{"arg":{"type":"string"}}}}}]"#;
    let tools_val = reference_jinja::parse_json_jval_bounded(tools_json, 65_536, 16)
        .unwrap_or_else(|_| JVal::array(Vec::new()));
    let tools_probe = render_with(
        JVal::array(vec![build_message_jval("user", "hello", use_typed)]),
        Some(tools_val.clone()),
    );
    let supports_tools = tools_probe.is_some_and(|bytes| {
        bytes
            .windows(b"caps_tool_alpha".len())
            .any(|win| win == b"caps_tool_alpha")
    });

    let tc_one_json = r#"[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"tc_probe_one","arguments":{"x":1}}}]}]"#;
    let tc_two_json = r#"[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"tc_probe_one","arguments":{"x":1}}},{"id":"call_2","type":"function","function":{"name":"tc_probe_two","arguments":{"y":2}}}]}]"#;
    let tc_one = reference_jinja::parse_json_jval_bounded(tc_one_json, 65_536, 16)
        .ok()
        .and_then(|msgs| render_with(msgs, Some(tools_val.clone())));
    let supports_tool_calls = tc_one.is_some_and(|bytes| {
        bytes
            .windows(b"tc_probe_one".len())
            .any(|win| win == b"tc_probe_one")
    });
    let tc_two = reference_jinja::parse_json_jval_bounded(tc_two_json, 65_536, 16)
        .ok()
        .and_then(|msgs| render_with(msgs, Some(tools_val)));
    let supports_parallel_tool_calls = tc_two.is_some_and(|bytes| {
        bytes
            .windows(b"tc_probe_one".len())
            .any(|win| win == b"tc_probe_one")
            && bytes
                .windows(b"tc_probe_two".len())
                .any(|win| win == b"tc_probe_two")
    });

    TemplateCaps {
        supports_tools,
        supports_tool_calls,
        supports_system_role,
        supports_parallel_tool_calls,
        supports_typed_content,
        supports_string_content,
    }
}

/// Clean-room implementation of `common_chat_templates_apply` for `common-chat-render`.
pub(super) fn common_apply_template(
    template: &str,
    bos: &str,
    eos: &str,
    messages: &[ChatMessage<'_>],
    add_generation_prompt: bool,
    use_jinja: bool,
    now_unix_seconds: Option<i64>,
) -> std::result::Result<String, String> {
    if !use_jinja {
        let effective = if template.is_empty() {
            "chatml"
        } else {
            template
        };
        return legacy_apply(effective, messages, add_generation_prompt)
            .ok_or_else(|| format!("unsupported legacy chat template: {effective}"));
    }

    let source = if template.is_empty() || template == "chatml" {
        CHATML_JINJA_TEMPLATE
    } else if !template.contains("{{")
        && !template.contains("{%")
        && format_from_name(template).is_some()
    {
        return legacy_apply(template, messages, add_generation_prompt)
            .ok_or_else(|| format!("unsupported built-in chat template: {template}"));
    } else {
        template
    };

    let program = parse_template_with_limit(source, MAX_TEMPLATE_BYTES)
        .map_err(|err| err.message().to_owned())?;
    let caps = template_caps_from_program(&program);
    let typed = !caps.supports_string_content && caps.supports_typed_content;
    let mut folded_msgs: Vec<(String, String)> = Vec::with_capacity(messages.len());
    if caps.supports_system_role {
        for msg in messages {
            folded_msgs.push((msg.role.to_owned(), msg.content.to_owned()));
        }
    } else {
        let mut pending_sys = String::new();
        for msg in messages {
            if msg.role == "system" {
                if !pending_sys.is_empty() {
                    pending_sys.push('\n');
                }
                pending_sys.push_str(msg.content);
            } else if msg.role == "user" && !pending_sys.is_empty() {
                let merged = if msg.content.is_empty() {
                    std::mem::take(&mut pending_sys)
                } else {
                    let prefix = std::mem::take(&mut pending_sys);
                    format!("{prefix}\n{}", msg.content)
                };
                folded_msgs.push((msg.role.to_owned(), merged));
            } else {
                folded_msgs.push((msg.role.to_owned(), msg.content.to_owned()));
            }
        }
        if !pending_sys.is_empty() {
            folded_msgs.insert(0, ("user".to_owned(), pending_sys));
        }
    }
    let msg_vals: Vec<JVal> = folded_msgs
        .iter()
        .map(|(role, content)| build_message_jval(role, content, typed))
        .collect();
    let globals = JVal::object(vec![
        (JVal::text_str("messages"), JVal::array(msg_vals)),
        (
            JVal::text_str("bos_token"),
            JVal::bytes(bos.as_bytes().to_vec(), true),
        ),
        (
            JVal::text_str("eos_token"),
            JVal::bytes(eos.as_bytes().to_vec(), true),
        ),
        (
            JVal::text_str("add_generation_prompt"),
            JVal::boolean(add_generation_prompt),
        ),
    ]);
    let now = now_unix_seconds.unwrap_or_else(current_unix_seconds);
    let spans = eval_program_with_now(&program, &globals, Some(now))
        .map_err(|err| err.message().to_owned())?;
    let bytes: Vec<u8> = spans.into_iter().flat_map(|s| s.bytes).collect();
    String::from_utf8(bytes).map_err(|_| "rendered chat template is not valid UTF-8".to_owned())
}

// ---------------------------------------------------------------------------
// Wire protocol validation and dispatch
// ---------------------------------------------------------------------------

fn require_exact_fields(obj: &Map<String, Value>, names: &[&str]) -> Result<()> {
    if obj.len() != names.len() || names.iter().any(|name| !obj.contains_key(*name)) {
        return Err(invalid("chat render fields mismatch"));
    }
    Ok(())
}

fn bounded_uint(val: &Value, maximum: usize) -> Result<usize> {
    let number = if let Some(u) = val.as_u64() {
        u
    } else if let Some(i) = val.as_i64() {
        u64::try_from(i).map_err(|_| invalid("chat integer must be nonnegative"))?
    } else {
        return Err(invalid("chat integer must be nonnegative"));
    };
    let max_u64 = u64::try_from(maximum)?;
    if number > max_u64 {
        return Err(invalid("chat integer exceeds bound"));
    }
    usize::try_from(number).map_err(Into::into)
}

fn bounded_text(val: &Value, maximum: usize) -> Result<&str> {
    let s = val
        .as_str()
        .ok_or_else(|| invalid("chat text must be string"))?;
    if s.len() > maximum || s.contains('\0') {
        return Err(invalid("chat text exceeds byte bound or contains NUL"));
    }
    Ok(s)
}

fn bounded_bool(val: &Value) -> Result<bool> {
    val.as_bool()
        .ok_or_else(|| invalid("chat boolean flag must be boolean"))
}

struct OwnedMessage {
    role: String,
    content: String,
}

enum ValidatedOp {
    Builtins {
        capacity: usize,
    },
    LegacyRender {
        template: String,
        messages: Vec<OwnedMessage>,
        add_generation: bool,
        capacity: usize,
    },
    CommonRender {
        template: String,
        messages: Vec<OwnedMessage>,
        add_generation: bool,
        bos: String,
        eos: String,
        use_jinja: bool,
    },
}

fn validate_messages(val: &Value) -> Result<Vec<OwnedMessage>> {
    let arr = val
        .as_array()
        .ok_or_else(|| invalid("chat Messages must be an array"))?;
    if arr.len() > MAX_MESSAGES {
        return Err(invalid("chat message count exceeds bound"));
    }
    let mut total = 0_usize;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let obj = entry
            .as_object()
            .ok_or_else(|| invalid("chat message must be an object"))?;
        require_exact_fields(obj, &["Role", "Content"])?;
        let role = bounded_text(
            obj.get("Role").ok_or_else(|| invalid("missing Role"))?,
            MAX_ROLE_BYTES,
        )?;
        let content = bounded_text(
            obj.get("Content")
                .ok_or_else(|| invalid("missing Content"))?,
            MAX_CONTENT_BYTES,
        )?;
        total = total
            .checked_add(role.len() + content.len())
            .ok_or_else(|| invalid("chat message byte total overflow"))?;
        if total > MAX_MESSAGES_TOTAL_BYTES {
            return Err(invalid("chat message byte total exceeds bound"));
        }
        out.push(OwnedMessage {
            role: role.to_owned(),
            content: content.to_owned(),
        });
    }
    Ok(out)
}

fn validate_operation(val: &Value) -> Result<ValidatedOp> {
    let obj = val
        .as_object()
        .ok_or_else(|| invalid("chat operation must be an object"))?;
    let kind = obj
        .get("Kind")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("chat operation Kind must be string"))?;
    match kind {
        "chat-builtins" => {
            require_exact_fields(obj, &["Kind", "Capacity"])?;
            let capacity = bounded_uint(
                obj.get("Capacity")
                    .ok_or_else(|| invalid("missing Capacity"))?,
                MAX_BUILTIN_CAPACITY,
            )?;
            Ok(ValidatedOp::Builtins { capacity })
        }
        "legacy-chat-render" => {
            require_exact_fields(
                obj,
                &["Kind", "Template", "Messages", "AddGeneration", "Capacity"],
            )?;
            let capacity = bounded_uint(
                obj.get("Capacity")
                    .ok_or_else(|| invalid("missing Capacity"))?,
                MAX_LEGACY_CAPACITY,
            )?;
            let template = bounded_text(
                obj.get("Template")
                    .ok_or_else(|| invalid("missing Template"))?,
                MAX_TEMPLATE_BYTES,
            )?
            .to_owned();
            let add_generation = bounded_bool(
                obj.get("AddGeneration")
                    .ok_or_else(|| invalid("missing AddGeneration"))?,
            )?;
            let messages = validate_messages(
                obj.get("Messages")
                    .ok_or_else(|| invalid("missing Messages"))?,
            )?;
            Ok(ValidatedOp::LegacyRender {
                template,
                messages,
                add_generation,
                capacity,
            })
        }
        "common-chat-render" => {
            require_exact_fields(
                obj,
                &[
                    "Kind",
                    "Template",
                    "Messages",
                    "AddGeneration",
                    "Bos",
                    "Eos",
                    "UseJinja",
                ],
            )?;
            let bos = bounded_text(
                obj.get("Bos").ok_or_else(|| invalid("missing Bos"))?,
                MAX_TOKEN_BYTES,
            )?
            .to_owned();
            let eos = bounded_text(
                obj.get("Eos").ok_or_else(|| invalid("missing Eos"))?,
                MAX_TOKEN_BYTES,
            )?
            .to_owned();
            let use_jinja = bounded_bool(
                obj.get("UseJinja")
                    .ok_or_else(|| invalid("missing UseJinja"))?,
            )?;
            let template = bounded_text(
                obj.get("Template")
                    .ok_or_else(|| invalid("missing Template"))?,
                MAX_TEMPLATE_BYTES,
            )?
            .to_owned();
            let add_generation = bounded_bool(
                obj.get("AddGeneration")
                    .ok_or_else(|| invalid("missing AddGeneration"))?,
            )?;
            let messages = validate_messages(
                obj.get("Messages")
                    .ok_or_else(|| invalid("missing Messages"))?,
            )?;
            Ok(ValidatedOp::CommonRender {
                template,
                messages,
                add_generation,
                bos,
                eos,
                use_jinja,
            })
        }
        _ => Err(invalid("unknown chat render kind")),
    }
}

fn execute_op(op: ValidatedOp) -> Result<Value> {
    match op {
        ValidatedOp::Builtins { capacity } => {
            let all = builtin_template_names();
            let names: Vec<Option<&str>> = (0..capacity).map(|idx| all.get(idx).copied()).collect();
            Ok(json!({
                "Count": all.len(),
                "Names": names,
            }))
        }
        ValidatedOp::LegacyRender {
            template,
            messages,
            add_generation,
            capacity,
        } => {
            let borrowed: Vec<ChatMessage<'_>> = messages
                .iter()
                .map(|m| ChatMessage {
                    role: &m.role,
                    content: &m.content,
                })
                .collect();
            let mut buffer = vec![0_u8; capacity];
            let return_code =
                if let Some(rendered) = legacy_apply(&template, &borrowed, add_generation) {
                    let raw = rendered.as_bytes();
                    let copy_len = raw.len().min(capacity);
                    buffer[..copy_len].copy_from_slice(&raw[..copy_len]);
                    i32::try_from(raw.len()).map_err(|_| invalid("legacy output exceeds i32"))?
                } else {
                    -1
                };
            Ok(json!({
                "ReturnCode": return_code,
                "BufferHex": hex::encode(buffer),
            }))
        }
        ValidatedOp::CommonRender {
            template,
            messages,
            add_generation,
            bos,
            eos,
            use_jinja,
        } => {
            let borrowed: Vec<ChatMessage<'_>> = messages
                .iter()
                .map(|m| ChatMessage {
                    role: &m.role,
                    content: &m.content,
                })
                .collect();
            Ok(
                match common_apply_template(
                    &template,
                    &bos,
                    &eos,
                    &borrowed,
                    add_generation,
                    use_jinja,
                    None,
                ) {
                    Ok(prompt) => json!({"Success": true, "Prompt": prompt, "Error": null}),
                    Err(err) => json!({"Success": false, "Prompt": null, "Error": err}),
                },
            )
        }
    }
}

/// Returns `true` when every operation in `request` belongs to `chat_templates`.
pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("chat-builtins" | "legacy-chat-render" | "common-chat-render")
                )
            })
    })
}

/// Validates and executes a bounded batch of `chat_templates` operations.
pub(super) fn execute(request: &Value) -> Result<Value> {
    let obj = request
        .as_object()
        .ok_or_else(|| invalid("chat template request must be a JSON object"))?;
    require_exact_fields(obj, &["SchemaVersion", "Operations"])?;
    let schema = obj
        .get("SchemaVersion")
        .ok_or_else(|| invalid("missing SchemaVersion"))?;
    if bounded_uint(schema, usize::MAX)? != 1 {
        return Err(invalid("unsupported SchemaVersion"));
    }
    let ops_arr = obj
        .get("Operations")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Operations must be an array"))?;
    if ops_arr.is_empty() || ops_arr.len() > MAX_OPERATIONS {
        return Err(invalid("Operations must contain 1 through 128 operations"));
    }
    let validated = ops_arr
        .iter()
        .map(validate_operation)
        .collect::<Result<Vec<_>>>()?;
    let results = validated
        .into_iter()
        .map(execute_op)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

/// Independent reference provider metadata for `tests/test-chat-template.cpp`.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-chat-templates",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA},
        "Architecture": "clean-room-legacy-template-detector-renderer-and-jinja-chat-adapter-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    fn wrap(op: &Value) -> Value {
        json!({"SchemaVersion": 1, "Operations": [op]})
    }

    #[test]
    fn independent_chat_templates_passes_all_73_source_conditions_across_reference_arms()
    -> Result<()> {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let report = crate::chat_template::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
        }
        assert!(crate::reference_provider::supports(ID));
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["AssertionSource"]["Sha256"], SOURCE_SHA);
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);

        let builtins_res = execute(&wrap(&json!({"Kind": "chat-builtins", "Capacity": 64})))?;
        let count = usize::try_from(
            builtins_res["Results"][0]["Count"]
                .as_u64()
                .ok_or_else(|| invalid("missing Count"))?,
        )?;
        assert_eq!(count, builtin_template_names().len());
        let names = builtins_res["Results"][0]["Names"]
            .as_array()
            .ok_or_else(|| invalid("missing Names"))?;
        assert_eq!(names.len(), 64);
        let non_null: Vec<&str> = names.iter().filter_map(Value::as_str).collect();
        assert_eq!(non_null.len(), count);
        let unique: std::collections::BTreeSet<&str> = non_null.iter().copied().collect();
        assert_eq!(unique.len(), count);
        for &builtin in builtin_template_names() {
            assert!(
                legacy_apply(
                    builtin,
                    &[ChatMessage {
                        role: "user",
                        content: "hi",
                    }],
                    true
                )
                .is_some(),
                "builtin {builtin} missing from legacy_apply"
            );
        }

        let prog = parse_template_with_limit(
            "{% for m in messages %}{{ m.role }}:{% for p in m.content | selectattr('type', 'equalto', 'text') %}{{ p.text }}{% endfor %}{% endfor %}{% if tools %}{{ tools[0].function.name }}{% endif %}",
            MAX_TEMPLATE_BYTES,
        )
        .map_err(|err| invalid(err.message()))?;
        let caps = template_caps_from_program(&prog);
        assert!(caps.supports_tools);
        assert!(!caps.supports_tool_calls);
        assert!(!caps.supports_parallel_tool_calls);
        assert!(caps.supports_system_role);
        assert!(caps.supports_typed_content);
        assert!(!caps.supports_string_content);

        let now_prog_src = "{{ strftime_now('%Y-%m-%d') }}:{{ messages[0].content }}";
        let rendered_now = common_apply_template(
            now_prog_src,
            "",
            "",
            &[ChatMessage {
                role: "user",
                content: "ok",
            }],
            false,
            true,
            Some(0),
        )
        .map_err(invalid)?;
        assert_eq!(rendered_now, "1970-01-01:ok");
        let rendered_legacy = common_apply_template(
            "",
            "",
            "",
            &[ChatMessage {
                role: "user",
                content: "hi",
            }],
            true,
            false,
            None,
        )
        .map_err(invalid)?;
        assert_eq!(
            rendered_legacy,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn independent_chat_templates_rejects_invalid_protocol_fields_and_bounds() {
        assert!(!handles(&json!({"Operations": []})));
        assert!(!handles(&wrap(&json!({"Kind": "jinja-caps"}))));

        for bad_req in [
            json!({"SchemaVersion": 2, "Operations": [{"Kind": "chat-builtins", "Capacity": 0}]}),
            json!({"SchemaVersion": 1, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": vec![json!({"Kind": "chat-builtins", "Capacity": 0}); 129]}),
            wrap(&json!({"Kind": "chat-builtins", "Capacity": 4097})),
            wrap(&json!({"Kind": "chat-builtins", "Capacity": -1})),
            wrap(&json!({"Kind": "chat-builtins", "Capacity": 0, "Extra": 1})),
            wrap(&json!({
                "Kind": "legacy-chat-render",
                "Template": "chatml\0",
                "Messages": [],
                "AddGeneration": true,
                "Capacity": 16
            })),
            wrap(&json!({
                "Kind": "legacy-chat-render",
                "Template": "chatml",
                "Messages": [],
                "AddGeneration": true,
                "Capacity": 131_073
            })),
            wrap(&json!({
                "Kind": "legacy-chat-render",
                "Template": "chatml",
                "Messages": [{"Role": "user", "Content": "hi", "Extra": 1}],
                "AddGeneration": true,
                "Capacity": 16
            })),
            wrap(&json!({
                "Kind": "common-chat-render",
                "Template": "chatml",
                "Messages": [],
                "AddGeneration": true,
                "Bos": "",
                "Eos": "",
                "UseJinja": "yes"
            })),
        ] {
            assert!(execute(&bad_req).is_err(), "{bad_req}");
        }
    }
}

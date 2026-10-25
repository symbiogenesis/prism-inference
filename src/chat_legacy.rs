//! Native text-only legacy chat formats, ported from pinned src/llama-chat.cpp.
use super::Message;
use crate::{Result, invalid};
use std::fmt::Write;
fn append(output: &mut String, args: std::fmt::Arguments<'_>) -> Result<()> {
    output
        .write_fmt(args)
        .map_err(|_| invalid("legacy chat formatting failed"))
}
pub(super) const NAMES: &[&str] = &[
    "bailing",
    "bailing-think",
    "bailing2",
    "chatglm3",
    "chatglm4",
    "chatml",
    "command-r",
    "deepseek",
    "deepseek-ocr",
    "deepseek2",
    "deepseek3",
    "exaone-moe",
    "exaone3",
    "exaone4",
    "falcon3",
    "gemma",
    "gigachat",
    "glmedge",
    "gpt-oss",
    "granite",
    "granite-4.0",
    "granite-4.1",
    "grok-2",
    "hunyuan-dense",
    "hunyuan-moe",
    "hunyuan-vl",
    "kimi-k2",
    "llama2",
    "llama2-sys",
    "llama2-sys-bos",
    "llama2-sys-strip",
    "llama3",
    "llama4",
    "megrez",
    "minicpm",
    "mistral-v1",
    "mistral-v3",
    "mistral-v3-tekken",
    "mistral-v7",
    "mistral-v7-tekken",
    "monarch",
    "openchat",
    "orion",
    "pangu-embedded",
    "phi3",
    "phi4",
    "rwkv-world",
    "seed_oss",
    "smolvlm",
    "solar-open",
    "vicuna",
    "vicuna-orca",
    "yandex",
    "zephyr",
];
fn trim(value: &str) -> &str {
    value.trim_matches(|ch: char| ch.is_ascii_whitespace())
}
#[allow(
    clippy::too_many_lines,
    reason = "Preserve the source template detector's ordered precedence checks"
)]
pub(super) fn detect(source: &str) -> Option<&'static str> {
    if let Some(name) = NAMES.iter().find(|&&name| name == source) {
        return Some(name);
    }
    let has = |needle: &str| source.contains(needle);
    Some(if has("<|im_start|>") {
        if has("<|im_sep|>") {
            "phi4"
        } else if has("<end_of_utterance>") {
            "smolvlm"
        } else {
            "chatml"
        }
    } else if source.starts_with("mistral") || has("[INST]") {
        if has("[SYSTEM_PROMPT]") {
            "mistral-v7"
        } else if has("' [INST] ' + system_message") || has("[AVAILABLE_TOOLS]") {
            if has(" [INST]") {
                "mistral-v1"
            } else if has("\"[INST]\"") {
                "mistral-v3-tekken"
            } else {
                "mistral-v3"
            }
        } else if has("content.strip()") {
            "llama2-sys-strip"
        } else if has("bos_token + '[INST]") {
            "llama2-sys-bos"
        } else if has("<<SYS>>") {
            "llama2-sys"
        } else {
            "llama2"
        }
    } else if has("<|assistant|>") && has("<|end|>") {
        "phi3"
    } else if has("[gMASK]<sop>") {
        "chatglm4"
    } else if has("<|assistant|>") && has("<|user|>") {
        if has("<|tool_declare|>") {
            "exaone-moe"
        } else if has("</s>") {
            "falcon3"
        } else {
            "glmedge"
        }
    } else if has("<|{{ item['role'] }}|>") && has("<|begin_of_image|>") {
        "glmedge"
    } else if has("<|user|>") && has("<|endoftext|>") {
        "zephyr"
    } else if has("bos_token + message['role']") {
        "monarch"
    } else if has("<start_of_turn>") {
        "gemma"
    } else if has("'\\n\\nAssistant: ' + eos_token") {
        "orion"
    } else if has("GPT4 Correct ") {
        "openchat"
    } else if has("USER: ") && has("ASSISTANT: ") {
        if has("SYSTEM: ") {
            "vicuna-orca"
        } else {
            "vicuna"
        }
    } else if has("### Instruction:") && has("<|EOT|>") {
        "deepseek"
    } else if has("<|START_OF_TURN_TOKEN|>") && has("<|USER_TOKEN|>") {
        "command-r"
    } else if has("<|start_header_id|>") && has("<|end_header_id|>") {
        "llama3"
    } else if has("[gMASK]sop") {
        "chatglm3"
    } else if has("<用户>") {
        "minicpm"
    } else if has("'Assistant: ' + message['content'] + eos_token") {
        "deepseek2"
    } else if has("<｜Assistant｜>") && has("<｜User｜>") && has("<｜end▁of▁sentence｜>")
    {
        "deepseek3"
    } else if has("[|system|]") && has("[|assistant|]") && has("[|endofturn|]") {
        if has("[|tool|]") {
            "exaone4"
        } else {
            "exaone3"
        }
    } else if has("rwkv-world") || has("{{- 'User: ' + message['content']|trim + '\\n\\n' -}}") {
        "rwkv-world"
    } else if has("<|start_of_role|>") {
        if has("<tool_call>") || has("<tools>") {
            if has("g4_default_system_message") {
                "granite-4.0"
            } else {
                "granite-4.1"
            }
        } else {
            "granite"
        }
    } else if has(
        "message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]",
    ) {
        "gigachat"
    } else if has("<|role_start|>") {
        "megrez"
    } else if has(" Ассистент:") {
        "yandex"
    } else if has("<role>ASSISTANT</role>") && has("'HUMAN'") {
        "bailing"
    } else if has("<role>ASSISTANT</role>") && has("\"HUMAN\"") && has("<think>") {
        "bailing-think"
    } else if has("<role>ASSISTANT</role>") && has("<role>HUMAN</role>") && has("<|role_end|>") {
        "bailing2"
    } else if has("<|header_start|>") && has("<|header_end|>") {
        "llama4"
    } else if has("<|endofuserprompt|>") {
        "dots1"
    } else if has("<|extra_0|>") && has("<|extra_4|>") {
        "hunyuan-moe"
    } else if has("<|start|>") && has("<|channel|>") {
        "gpt-oss"
    } else if has("<｜hy_Assistant｜>") && has("<｜hy_begin▁of▁sentence｜>") {
        "hunyuan-vl"
    } else if has("<｜hy_Assistant｜>") && has("<｜hy_place▁holder▁no▁3｜>") {
        "hunyuan-dense"
    } else if has("<|im_assistant|>assistant<|im_middle|>") {
        "kimi-k2"
    } else if has("<seed:bos>") {
        "seed_oss"
    } else if has("'Assistant: '  + message['content'] + '<|separator|>") {
        "grok-2"
    } else if has("[unused9]系统：[unused10]") {
        "pangu-embedded"
    } else if has("<|begin|>") && has("<|end|>") && has("<|content|>") {
        "solar-open"
    } else {
        return None;
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "Keep the source legacy template format dispatch and stateful turn transitions together"
)]
pub(super) fn render(source: &str, messages: &[Message], add: bool) -> Result<String> {
    let format = detect(source).ok_or_else(|| invalid("unsupported legacy chat template"))?;
    if messages.len() > 1024
        || messages
            .iter()
            .map(|message| message.role.len() + message.content.len())
            .sum::<usize>()
            > 1024 * 1024
    {
        return Err(invalid("chat message bound exceeded"));
    }
    let mut out = String::new();
    match format {
        "llama2" | "llama2-sys" | "llama2-sys-bos" | "llama2-sys-strip" => {
            let mut inside = true;
            out.push_str("[INST] ");
            for message in messages {
                if !inside {
                    inside = true;
                    out.push_str(if format == "llama2-sys-bos" {
                        "<s>[INST] "
                    } else {
                        "[INST] "
                    });
                }
                let content = if format == "llama2-sys-strip" {
                    trim(&message.content)
                } else {
                    &message.content
                };
                match message.role.as_str() {
                    "system" => {
                        if format == "llama2" {
                            out.push_str(content);
                            out.push('\n');
                        } else {
                            out.push_str("<<SYS>>\n");
                            out.push_str(content);
                            out.push_str("\n<</SYS>>\n\n");
                        }
                    }
                    "user" => {
                        out.push_str(content);
                        out.push_str(" [/INST]");
                    }
                    _ => {
                        out.push_str(content);
                        out.push_str("</s>");
                        inside = false;
                    }
                }
            }
        }
        "mistral-v1" | "mistral-v3" | "mistral-v3-tekken" => {
            let lead = if format == "mistral-v1" { " " } else { "" };
            let trail = if format == "mistral-v3-tekken" {
                ""
            } else {
                " "
            };
            let mut inside = false;
            for message in messages {
                if !inside {
                    out.push_str(lead);
                    out.push_str("[INST]");
                    out.push_str(trail);
                    inside = true;
                }
                match message.role.as_str() {
                    "system" => {
                        out.push_str(&message.content);
                        out.push_str("\n\n");
                    }
                    "user" => {
                        out.push_str(&message.content);
                        out.push_str(lead);
                        out.push_str("[/INST]");
                    }
                    _ => {
                        out.push_str(trail);
                        out.push_str(if format == "mistral-v3" {
                            trim(&message.content)
                        } else {
                            &message.content
                        });
                        out.push_str("</s>");
                        inside = false;
                    }
                }
            }
        }
        "mistral-v7" | "mistral-v7-tekken" => {
            let space = if format == "mistral-v7" { " " } else { "" };
            for message in messages {
                let (start, end) = match message.role.as_str() {
                    "system" => ("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"),
                    "user" => ("[INST]", "[/INST]"),
                    _ => ("", "</s>"),
                };
                out.push_str(start);
                out.push_str(space);
                out.push_str(&message.content);
                out.push_str(end);
            }
        }
        "gemma" | "orion" => {
            let mut system = String::new();
            for message in messages {
                if message.role == "system" {
                    system.push_str(if format == "gemma" {
                        trim(&message.content)
                    } else {
                        &message.content
                    });
                    continue;
                }
                if format == "gemma" {
                    let role = if message.role == "assistant" {
                        "model"
                    } else {
                        &message.role
                    };
                    out.push_str("<start_of_turn>");
                    out.push_str(role);
                    out.push('\n');
                    if !system.is_empty() && role != "model" {
                        out.push_str(&system);
                        out.push_str("\n\n");
                        system.clear();
                    }
                    out.push_str(trim(&message.content));
                    out.push_str("<end_of_turn>\n");
                } else if message.role == "user" {
                    out.push_str("Human: ");
                    if !system.is_empty() {
                        out.push_str(&system);
                        out.push_str("\n\n");
                        system.clear();
                    }
                    out.push_str(&message.content);
                    out.push_str("\n\nAssistant: </s>");
                } else {
                    out.push_str(&message.content);
                    out.push_str("</s>");
                }
            }
            if add && format == "gemma" {
                out.push_str("<start_of_turn>model\n");
            }
        }
        "gigachat" => {
            let system = messages
                .first()
                .is_some_and(|message| message.role == "system");
            out.push_str("<s>");
            if system {
                out.push_str(&messages[0].content);
                out.push_str("<|message_sep|>");
            }
            for message in messages.iter().skip(usize::from(system)) {
                if message.role == "user" {
                    out.push_str("user<|role_sep|>");
                    out.push_str(&message.content);
                    out.push_str("<|message_sep|>available functions<|role_sep|>[]<|message_sep|>");
                } else if message.role == "assistant" {
                    out.push_str("assistant<|role_sep|>");
                    out.push_str(&message.content);
                    out.push_str("<|message_sep|>");
                }
            }
            if add {
                out.push_str("assistant<|role_sep|>");
            }
        }
        _ => {
            if format == "chatglm3" {
                out.push_str("[gMASK]sop");
            }
            if format == "chatglm4" {
                out.push_str("[gMASK]<sop>");
            }
            if format == "smolvlm" {
                out.push_str("<|im_start|>");
            }
            if format == "hunyuan-vl" {
                out.push_str("<｜hy_begin▁of▁sentence｜>");
            }
            if format == "bailing2"
                && messages
                    .first()
                    .is_none_or(|message| message.role != "system")
            {
                out.push_str("<role>SYSTEM</role>detailed thinking off<|role_end|>");
            }
            for (index, message) in messages.iter().enumerate() {
                let role = message.role.as_str();
                let content = message.content.as_str();
                match format {
                    "chatml" => append(
                        &mut out,
                        format_args!("<|im_start|>{role}\n{content}<|im_end|>\n"),
                    )?,
                    "phi3" => append(&mut out, format_args!("<|{role}|>\n{content}<|end|>\n"))?,
                    "phi4" => {
                        append(
                            &mut out,
                            format_args!("<|im_start|>{role}<|im_sep|>{content}<|im_end|>"),
                        )?;
                    }
                    "falcon3" => append(&mut out, format_args!("<|{role}|>\n{content}\n"))?,
                    "zephyr" => append(
                        &mut out,
                        format_args!("<|{role}|>\n{content}<|endoftext|>\n"),
                    )?,
                    "monarch" => append(
                        &mut out,
                        format_args!(
                            "{}{role}\n{content}</s>\n",
                            if index == 0 { "" } else { "<s>" }
                        ),
                    )?,
                    "openchat" => {
                        if role == "system" {
                            out.push_str(content);
                            out.push_str("<|end_of_turn|>");
                        } else {
                            let mut role = role.to_owned();
                            if let Some(first) = role.get_mut(..1) {
                                first.make_ascii_uppercase();
                            }
                            append(
                                &mut out,
                                format_args!("GPT4 Correct {role}: {content}<|end_of_turn|>"),
                            )?;
                        }
                    }
                    "vicuna" | "vicuna-orca" => match role {
                        "system" => {
                            if format == "vicuna-orca" {
                                append(&mut out, format_args!("SYSTEM: {content}\n"))?;
                            } else {
                                append(&mut out, format_args!("{content}\n\n"))?;
                            }
                        }
                        "user" => append(&mut out, format_args!("USER: {content}\n"))?,
                        "assistant" => {
                            append(&mut out, format_args!("ASSISTANT: {content}</s>\n"))?;
                        }
                        _ => {}
                    },
                    "deepseek" => match role {
                        "system" => out.push_str(content),
                        "user" => append(&mut out, format_args!("### Instruction:\n{content}\n"))?,
                        "assistant" => {
                            append(
                                &mut out,
                                format_args!("### Response:\n{content}\n<|EOT|>\n"),
                            )?;
                        }
                        _ => {}
                    },
                    "command-r" => {
                        if let Some(token) = match role {
                            "system" => Some("SYSTEM"),
                            "user" => Some("USER"),
                            "assistant" => Some("CHATBOT"),
                            _ => None,
                        } {
                            append(
                                &mut out,
                                format_args!(
                                    "<|START_OF_TURN_TOKEN|><|{token}_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
                                    trim(content)
                                ),
                            )?;
                        }
                    }
                    "llama3" => append(
                        &mut out,
                        format_args!(
                            "<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>",
                            trim(content)
                        ),
                    )?,
                    "llama4" => append(
                        &mut out,
                        format_args!(
                            "<|header_start|>{role}<|header_end|>\n\n{}<|eot|>",
                            trim(content)
                        ),
                    )?,
                    "chatglm3" => append(&mut out, format_args!("<|{role}|>\n {content}"))?,
                    "chatglm4" | "glmedge" => {
                        append(&mut out, format_args!("<|{role}|>\n{content}"))?;
                    }
                    "minicpm" => {
                        if role == "user" {
                            out.push_str("<用户>");
                        }
                        out.push_str(trim(content));
                        if role == "user" {
                            out.push_str("<AI>");
                        }
                    }
                    "deepseek2" | "deepseek3" => match role {
                        "system" => append(&mut out, format_args!("{content}\n\n"))?,
                        "user" => {
                            if format == "deepseek2" {
                                append(&mut out, format_args!("User: {content}\n\n"))?;
                            } else {
                                append(&mut out, format_args!("<｜User｜>{content}"))?;
                            }
                        }
                        "assistant" => append(
                            &mut out,
                            format_args!(
                                "{}{content}<｜end▁of▁sentence｜>",
                                if format == "deepseek2" {
                                    "Assistant: "
                                } else {
                                    "<｜Assistant｜>"
                                }
                            ),
                        )?,
                        _ => {}
                    },
                    "deepseek-ocr" => out.push_str(content),
                    "exaone3" | "exaone4" => {
                        if matches!(role, "system" | "user" | "assistant")
                            || format == "exaone4" && role == "tool"
                        {
                            append(
                                &mut out,
                                format_args!(
                                    "[|{role}|]{}{}\n",
                                    trim(content),
                                    if role == "user" { "" } else { "[|endofturn|]" }
                                ),
                            )?;
                        }
                    }
                    "exaone-moe" => {
                        if matches!(role, "system" | "user" | "assistant" | "tool") {
                            append(
                                &mut out,
                                format_args!("<|{role}|>\n{}<|endofturn|>\n", trim(content)),
                            )?;
                        }
                    }
                    "rwkv-world" => {
                        if let Some(prefix) = match role {
                            "system" => Some("System"),
                            "user" => Some("User"),
                            "assistant" => Some("Assistant"),
                            _ => None,
                        } {
                            append(&mut out, format_args!("{prefix}: {}\n\n", trim(content)))?;
                            if role == "user" && index + 1 == messages.len() {
                                out.push_str("Assistant:");
                            }
                        }
                    }
                    "granite" | "granite-4.0" | "granite-4.1" => {
                        let display = if role == "assistant_tool_call" && format != "granite" {
                            "assistant"
                        } else {
                            role
                        };
                        append(
                            &mut out,
                            format_args!(
                                "<|start_of_role|>{display}<|end_of_role|>{}{content}<|end_of_text|>\n",
                                if role == "assistant_tool_call" {
                                    "<|tool_call|>"
                                } else {
                                    ""
                                }
                            ),
                        )?;
                    }
                    "megrez" => append(
                        &mut out,
                        format_args!("<|role_start|>{role}<|role_end|>{content}<|turn_end|>"),
                    )?,
                    "yandex" => {
                        if let Some(prefix) = match role {
                            "user" => Some(" Пользователь"),
                            "assistant" => Some(" Ассистент"),
                            _ => None,
                        } {
                            append(&mut out, format_args!("{prefix}: {content}\n\n"))?;
                        }
                    }
                    "bailing" | "bailing-think" | "bailing2" => {
                        let display = if role == "user" {
                            "HUMAN".into()
                        } else {
                            role.to_ascii_uppercase()
                        };
                        append(
                            &mut out,
                            format_args!(
                                "<role>{display}</role>{content}{}",
                                if format == "bailing2" {
                                    "<|role_end|>"
                                } else {
                                    ""
                                }
                            ),
                        )?;
                    }
                    "smolvlm" => match role {
                        "system" => append(&mut out, format_args!("{content}\n\n"))?,
                        "user" => append(
                            &mut out,
                            format_args!("User: {content}<end_of_utterance>\n"),
                        )?,
                        _ => append(
                            &mut out,
                            format_args!("Assistant: {content}<end_of_utterance>\n"),
                        )?,
                    },
                    "dots1" => {
                        let (start, end) = match role {
                            "system" => ("system", "endofsystem"),
                            "user" => ("userprompt", "endofuserprompt"),
                            _ => ("response", "endofresponse"),
                        };
                        append(&mut out, format_args!("<|{start}|>{content}<|{end}|>"))?;
                    }
                    "hunyuan-moe" => match role {
                        "system" => append(
                            &mut out,
                            format_args!("<|startoftext|>{content}<|extra_4|>"),
                        )?,
                        "assistant" => append(&mut out, format_args!("{content}<|eos|>"))?,
                        _ => append(
                            &mut out,
                            format_args!("<|startoftext|>{content}<|extra_0|>"),
                        )?,
                    },
                    "gpt-oss" => append(
                        &mut out,
                        format_args!(
                            "<|start|>{role}<|message|>{content}{}",
                            if role == "assistant" {
                                "<|return|>"
                            } else {
                                "<|end|>"
                            }
                        ),
                    )?,
                    "hunyuan-dense" => {
                        if index == 0 && role == "system" {
                            append(
                                &mut out,
                                format_args!("{content}<｜hy_place▁holder▁no▁3｜>"),
                            )?;
                        }
                        if role == "assistant" {
                            append(
                                &mut out,
                                format_args!(
                                    "<｜hy_Assistant｜>{content}<｜hy_place▁holder▁no▁2｜>"
                                ),
                            )?;
                        } else if role == "user" {
                            append(
                                &mut out,
                                format_args!("<｜hy_User｜>{content}<｜hy_Assistant｜>"),
                            )?;
                        }
                    }
                    "hunyuan-vl" => {
                        if index == 0 && role == "system" {
                            append(
                                &mut out,
                                format_args!("{content}<｜hy_place▁holder▁no▁3｜>"),
                            )?;
                        } else if role == "user" {
                            append(&mut out, format_args!("{content}<｜hy_User｜>"))?;
                        } else if role == "assistant" {
                            append(&mut out, format_args!("{content}<｜hy_Assistant｜>"))?;
                        }
                    }
                    "kimi-k2" => {
                        if let Some(prefix) = match role {
                            "system" | "tool" => Some("system"),
                            "user" => Some("user"),
                            "assistant" => Some("assistant"),
                            _ => None,
                        } {
                            append(&mut out, format_args!("<|im_{prefix}|>{role}<|im_middle|>"))?;
                        }
                        append(&mut out, format_args!("{content}<|im_end|>"))?;
                    }
                    "seed_oss" => append(
                        &mut out,
                        format_args!(
                            "<seed:bos>{role}\n{}<seed:eos>",
                            if role == "assistant" {
                                trim(content)
                            } else {
                                content
                            }
                        ),
                    )?,
                    "grok-2" => {
                        if let Some(prefix) = match role {
                            "system" => Some("System"),
                            "user" => Some("Human"),
                            "assistant" => Some("Assistant"),
                            _ => None,
                        } {
                            append(
                                &mut out,
                                format_args!(
                                    "{prefix}: {}<|separator|>\n\n",
                                    if role == "assistant" {
                                        content
                                    } else {
                                        trim(content)
                                    }
                                ),
                            )?;
                        }
                    }
                    "pangu-embedded" => {
                        if index == 0 && role != "system" {
                            out.push_str("[unused9]系统：[unused10]");
                        }
                        if let Some(prefix) = match role {
                            "system" => Some("系统"),
                            "user" => Some("用户"),
                            "assistant" => Some("助手"),
                            "tool" => Some("工具"),
                            "function" => Some("方法"),
                            _ => None,
                        } {
                            append(
                                &mut out,
                                format_args!("[unused9]{prefix}：{content}[unused10]"),
                            )?;
                        }
                    }
                    "solar-open" => {
                        append(
                            &mut out,
                            format_args!("<|begin|>{role}<|content|>{content}<|end|>"),
                        )?;
                    }
                    _ => return Err(invalid("unsupported legacy chat formatter")),
                }
            }
            if add {
                out.push_str(match format {
                    "chatml" => "<|im_start|>assistant\n",
                    "phi3" | "falcon3" | "zephyr" | "exaone-moe" | "chatglm4" => "<|assistant|>\n",
                    "phi4" => "<|im_start|>assistant<|im_sep|>",
                    "monarch" => "<s>assistant\n",
                    "openchat" => "GPT4 Correct Assistant:",
                    "vicuna" | "vicuna-orca" => "ASSISTANT:",
                    "deepseek" => "### Response:\n",
                    "command-r" => "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
                    "llama3" => "<|start_header_id|>assistant<|end_header_id|>\n\n",
                    "llama4" => "<|header_start|>assistant<|header_end|>\n\n",
                    "chatglm3" | "glmedge" => "<|assistant|>",
                    "deepseek2" | "smolvlm" | "grok-2" => "Assistant:",
                    "deepseek3" => "<｜Assistant｜>",
                    "exaone3" | "exaone4" => "[|assistant|]",
                    "granite" | "granite-4.0" | "granite-4.1" => {
                        "<|start_of_role|>assistant<|end_of_role|>"
                    }
                    "megrez" => "<|role_start|>assistant<|role_end|>",
                    "yandex" => " Ассистент:[SEP]",
                    "bailing" | "bailing2" => "<role>ASSISTANT</role>",
                    "bailing-think" => "<role>ASSISTANT</role><think>",
                    "dots1" => "<|response|>",
                    "gpt-oss" => "<|start|>assistant",
                    "kimi-k2" => "<|im_assistant|>assistant<|im_middle|>",
                    "seed_oss" => "<seed:bos>assistant\n",
                    "pangu-embedded" => "[unused9]助手：",
                    "solar-open" => "<|begin|>assistant",
                    _ => "",
                });
            }
        }
    }
    if out.len() > 4 * 1024 * 1024 {
        return Err(invalid("legacy chat output bound exceeded"));
    }
    Ok(out)
}

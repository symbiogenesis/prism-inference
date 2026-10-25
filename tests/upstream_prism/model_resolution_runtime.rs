//! Thin native/Rust observation adapters for production model-resolution APIs.

use super::super::engine::Engine;
use super::super::model_session::{Client as NativeClient, ModelSession as NativeSession};
use super::super::{Result, invalid};
use bonsai_prism_runtime::resolution::{File, Handler, Model, Options, Params, Plan, Resolver};
use serde_json::{Value, json};
use std::path::PathBuf;

pub(super) enum Client {
    Native(NativeClient),
    Runtime(Resolver),
}
pub(super) enum ModelSession {
    Native(NativeSession),
    Runtime(Resolver),
}
impl ModelSession {
    pub(super) fn open_with_environment(
        engine: &Engine,
        name: &str,
        environment: &[(String, String)],
    ) -> Result<Option<Self>> {
        if let Some(session) = NativeSession::open_with_environment(engine, name, environment)? {
            return Ok(Some(Self::Native(session)));
        }
        let get = |key| {
            environment
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.clone())
                .ok_or_else(|| invalid(format!("resolution environment lacks {key}")))
        };
        let resolver = Resolver::new(PathBuf::from(get("LLAMA_CACHE")?), get("MODEL_ENDPOINT")?)?;
        Ok(Some(Self::Runtime(resolver)))
    }
    pub(super) fn client(&self) -> Client {
        match self {
            Self::Native(session) => Client::Native(session.client()),
            Self::Runtime(resolver) => Client::Runtime(resolver.clone()),
        }
    }
    pub(super) fn finish(self) -> Result<()> {
        match self {
            Self::Native(session) => session.finish(),
            Self::Runtime(_) => Ok(()),
        }
    }
}
fn file(file: &File) -> Value {
    json!({"Path":file.path,"Url":file.url,"LocalPath":file.local_path,"FinalPath":file.final_path,"Oid":file.oid,"Repo":file.repo})
}
fn plan(value: &Plan) -> Value {
    json!({"Primary":file(&value.primary),"ModelFiles":value.model_files.iter().map(file).collect::<Vec<_>>(),"Mmproj":file(&value.mmproj),"Mtp":file(&value.mtp),"Dflash":file(&value.dflash),"Eagle3":file(&value.eagle3),"Dspark":file(&value.dspark),"Preset":file(&value.preset)})
}
#[allow(
    clippy::as_conversions,
    reason = "Serialize the explicit repr(u32) speculative policy enum into its native wire identifier."
)]
fn snapshot(value: &Params) -> Value {
    json!({"ModelPath":value.model.path,"ModelHfRepo":value.model.hf_repo,"MmprojPath":value.mmproj.path,"DraftPath":value.draft.path,"SpecTypes":value.spec_types.iter().map(|v| *v as u32).collect::<Vec<_>>(),"Preset":value.preset,"Offline":value.offline})
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| invalid(format!("resolution string {key} missing")))
}
fn flag(value: &Value, key: &str) -> Result<bool> {
    value[key]
        .as_bool()
        .ok_or_else(|| invalid(format!("resolution boolean {key} missing")))
}
impl Client {
    pub(super) fn invoke(&self, operation: Value) -> Result<Value> {
        let Self::Runtime(resolver) = self else {
            let Self::Native(client) = self else {
                return Err(invalid("resolution client variant"));
            };
            return Ok(client.call(0, operation, None)?.value);
        };
        match text(&operation, "Op")? {
            "resolution-pause-log" => Ok(Value::Null),
            "resolution-plan" => {
                let model = Model {
                    hf_repo: text(&operation, "HfRepo")?.into(),
                    hf_file: text(&operation, "HfFile")?.into(),
                    ..Model::default()
                };
                let value = &operation["Options"];
                let options = Options {
                    mmproj: flag(value, "Mmproj")?,
                    mtp: flag(value, "Mtp")?,
                    dflash: flag(value, "Dflash")?,
                    eagle3: flag(value, "Eagle3")?,
                    dspark: flag(value, "Dspark")?,
                    ..Options::default()
                };
                Ok(plan(&resolver.plan(&model, &options)?))
            }
            "resolution-handler" => {
                let arguments = operation["Arguments"]
                    .as_array()
                    .ok_or_else(|| invalid("missing arguments"))?
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| invalid("argument must be string"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let Ok(mut params) = Params::parse(&arguments) else {
                    return Ok(
                        json!({"Parsed":false,"Params":snapshot(&Params::default()),"Plan":null,"DraftPlan":null}),
                    );
                };
                let example = u32::try_from(
                    operation["Example"]
                        .as_u64()
                        .ok_or_else(|| invalid("invalid example"))?,
                )?;
                let mut handler = Handler::initialize(resolver, &params, example)?;
                params.offline = flag(&operation, "OfflineAfterInit")?;
                if flag(&operation, "Apply")? {
                    handler.apply(resolver, &mut params)?;
                }
                Ok(
                    json!({"Parsed":true,"Params":snapshot(&params),"Plan":plan(&handler.plan),"DraftPlan":plan(&handler.draft_plan)}),
                )
            }
            _ => Err(invalid("unknown resolution operation")),
        }
    }
}

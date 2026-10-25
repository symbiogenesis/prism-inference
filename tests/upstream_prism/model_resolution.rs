//! Full safe Rust HF model-resolution and handler-assembly source contracts.
//!
//! Rust serves the original synthetic HF listings over bounded loopback HTTP.
//! The selected production runtime resolves listings, parses model options and owns its cache.
//! Pinned source b87ca7225456b06b0975b8239c66b5aecf07e37c, MIT.

use super::engine::Engine;
#[path = "model_resolution_runtime.rs"]
mod runtime;
use super::{Result, invalid, result};
use runtime::{Client, ModelSession};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

#[path = "model_resolution_server.rs"]
mod server;
use server::{FixtureServer, Repositories};

const ID: &str = "tests/test-model-resolution.cpp";
const SOURCE_SHA: &str = "d596a6befca98035acfab4e71e72f50a698a11062fcac138735b3fe1aa9af593";
const INPUTS: &str = include_str!("model-resolution-inputs.json");
const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[derive(Default)]
struct Checks {
    source: usize,
    guards: usize,
    failed: Vec<String>,
}
impl Checks {
    fn source(&mut self, passed: bool, message: impl Into<String>) {
        self.source += 1;
        if !passed && self.failed.len() < 32 {
            self.failed.push(message.into());
        }
    }
    fn guard(&mut self, passed: bool, message: &str) -> Result<()> {
        self.guards += 1;
        verify(passed, message)
    }
}
fn verify(passed: bool, message: &str) -> Result<()> {
    if passed {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| invalid(format!("missing resolution string {key}")))
}
fn strings(value: &Value) -> Result<Vec<String>> {
    value
        .as_array()
        .ok_or_else(|| invalid("missing resolution list"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("nonstring resolution list entry"))
        })
        .collect()
}
fn invoke(client: &Client, operation: Value) -> Result<Value> {
    client.invoke(operation)
}
fn cached(cache: &Path, repo: &str, file: &str) -> Result<String> {
    cache
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(COMMIT)
        .join(file)
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid("cache path is not UTF-8"))
}
fn layouts(fixture: &Value) -> Result<Repositories> {
    fixture["Layouts"]
        .as_object()
        .ok_or_else(|| invalid("missing source listing layouts"))?
        .iter()
        .map(|(name, files)| Ok((name.clone(), strings(files)?)))
        .collect()
}
fn plan_predicates(value: &Value, case: &Value, checks: &mut Checks) -> Result<()> {
    for key in ["Primary", "Mmproj", "Mtp", "Dflash", "Eagle3", "Dspark"] {
        let actual = text(&value[key], "Path")?;
        checks.source(
            actual == text(case, key)?,
            format!("{}: {key} resolution differs", text(case, "Name")?),
        );
    }
    let mut actual = value["ModelFiles"]
        .as_array()
        .ok_or_else(|| invalid("missing actual model files"))?
        .iter()
        .map(|file| text(file, "Path").map(str::to_owned))
        .collect::<Result<Vec<_>>>()?;
    let mut expected = strings(&case["ModelFiles"])?;
    actual.sort_unstable();
    expected.sort_unstable();
    checks.source(
        actual == expected,
        format!("{}: shard set differs", text(case, "Name")?),
    );
    if let Some(first) = expected.first() {
        checks.source(
            text(&value["Primary"], "Path")? == first,
            format!("{}: primary is not first split", text(case, "Name")?),
        );
    }
    Ok(())
}
fn plans(
    client: &Client,
    server: &FixtureServer,
    fixture: &Value,
    listings: &Repositories,
    checks: &mut Checks,
) -> Result<usize> {
    let cases = fixture["PlanCases"]
        .as_array()
        .ok_or_else(|| invalid("missing plan source cases"))?;
    verify(cases.len() == 14, "source plan inventory changed")?;
    let mut variants = 0;
    for case in cases {
        let original = listings
            .get(text(case, "Layout")?)
            .ok_or_else(|| invalid("unknown source listing layout"))?;
        let dependent = case["OrderDependent"]
            .as_bool()
            .ok_or_else(|| invalid("missing source order-dependence flag"))?;
        let sidecars = case["Sidecars"]
            .as_bool()
            .ok_or_else(|| invalid("missing source sidecar flag"))?;
        for rotation in 0..original.len() {
            if dependent && rotation > 0 {
                continue;
            }
            let mut files = original.clone();
            files.rotate_left(rotation);
            if rotation % 2 == 1 {
                files.reverse();
            }
            server.replace(BTreeMap::from([("test/repo".into(), files)]))?;
            let before = server.request_count()?;
            let value = invoke(
                client,
                json!({"Op":"resolution-plan","HfRepo":case["HfRepo"],"HfFile":case["HfFile"],"Options":{"Mmproj":sidecars,"Mtp":sidecars,"Dflash":sidecars,"Eagle3":sidecars,"Dspark":sidecars}}),
            )?;
            checks.guard(
                server.request_count()?.saturating_sub(before) >= 2,
                "resolution did not exercise live refs/tree HTTP",
            )?;
            plan_predicates(&value, case, checks)?;
            variants += 1;
        }
    }
    checks.guard(
        variants == 94 && checks.source == 721,
        "original94plan variants/721predicates changed",
    )?;
    server.replace(BTreeMap::new())?;
    Ok(variants)
}
struct Assembly {
    arguments: Vec<&'static str>,
    expected: Vec<(&'static str, Value)>,
}
const fn assembly(arguments: Vec<&'static str>, expected: Vec<(&'static str, Value)>) -> Assembly {
    Assembly {
        arguments,
        expected,
    }
}
fn assembly_cases(cache: &Path) -> Result<Vec<Assembly>> {
    let mut cases = basic_assembly_cases(cache)?;
    cases.extend(speculative_assembly_cases(cache)?);
    Ok(cases)
}
fn basic_assembly_cases(cache: &Path) -> Result<Vec<Assembly>> {
    Ok(vec![
        assembly(
            vec!["server", "-hf", "test/main:Q8_0"],
            vec![
                (
                    "ModelPath",
                    json!(cached(cache, "test/main", "model-Q8_0.gguf")?),
                ),
                (
                    "MmprojPath",
                    json!(cached(cache, "test/main", "mmproj-model-Q8_0.gguf")?),
                ),
                ("DraftPath", json!("")),
            ],
        ),
        assembly(
            vec!["server", "-hf", "test/main:Q8_0", "--no-mmproj"],
            vec![("MmprojPath", json!(""))],
        ),
        assembly(
            vec![
                "server",
                "-hf",
                "test/main:Q8_0",
                "--mmproj",
                "/local/mmproj.gguf",
            ],
            vec![("MmprojPath", json!("/local/mmproj.gguf"))],
        ),
        assembly(
            vec![
                "server",
                "-hf",
                "test/main:Q8_0",
                "--spec-type",
                "draft-mtp",
            ],
            vec![(
                "DraftPath",
                json!(cached(cache, "test/main", "mtp-model-Q8_0.gguf")?),
            )],
        ),
        assembly(
            vec![
                "server",
                "-hf",
                "test/hole:Q8_0",
                "-hfd",
                "test/hole:Q4_0",
                "--spec-type",
                "draft-mtp",
            ],
            vec![(
                "DraftPath",
                json!(cached(cache, "test/hole", "mtp-model-Q4_0.gguf")?),
            )],
        ),
        assembly(
            vec![
                "server",
                "-hf",
                "test/main:Q8_0",
                "-hfd",
                "test/main",
                "-md",
                "mtp-model-BF16.gguf",
                "--spec-type",
                "draft-mtp",
            ],
            vec![(
                "DraftPath",
                json!(cached(cache, "test/main", "mtp-model-BF16.gguf")?),
            )],
        ),
    ])
}
fn speculative_assembly_cases(cache: &Path) -> Result<Vec<Assembly>> {
    Ok(vec![
        assembly(
            vec!["server", "-hf", "test/main:Q8_0", "-hfd", "test/quad:Q8_0"],
            vec![
                ("SpecTypes", json!([3])),
                (
                    "DraftPath",
                    json!(cached(cache, "test/quad", "mtp-model-Q8_0.gguf")?),
                ),
            ],
        ),
        assembly(
            vec![
                "server",
                "-hf",
                "test/main:Q8_0",
                "-hfd",
                "test/dflash:Q8_0",
            ],
            vec![
                ("SpecTypes", json!([4])),
                (
                    "DraftPath",
                    json!(cached(cache, "test/dflash", "dflash-model-Q8_0.gguf")?),
                ),
            ],
        ),
        assembly(
            vec![
                "server",
                "-hf",
                "test/main:Q8_0",
                "-hfd",
                "test/eagle3:Q8_0",
            ],
            vec![
                ("SpecTypes", json!([2])),
                (
                    "DraftPath",
                    json!(cached(cache, "test/eagle3", "eagle3-model-Q8_0.gguf")?),
                ),
            ],
        ),
        assembly(
            vec!["server", "-hf", "test/main:Q8_0", "-hfd", "test/pair:Q8_0"],
            vec![
                ("SpecTypes", json!([5])),
                (
                    "DraftPath",
                    json!(cached(cache, "test/pair", "dspark-model-Q8_0.gguf")?),
                ),
            ],
        ),
        assembly(
            vec!["server", "-hf", "test/spark", "--spec-type", "draft-dspark"],
            vec![
                (
                    "ModelPath",
                    json!(cached(cache, "test/spark", "model-MXFP4.gguf")?),
                ),
                (
                    "DraftPath",
                    json!(cached(cache, "test/spark", "dspark-model-MXFP4.gguf")?),
                ),
            ],
        ),
        assembly(
            vec!["server", "-hf", "test/main:Q8_0", "-hfd", "test/small"],
            vec![
                ("SpecTypes", json!([0])),
                (
                    "DraftPath",
                    json!(cached(cache, "test/small", "draft-model-Q4_K_M.gguf")?),
                ),
            ],
        ),
        assembly(
            vec!["server", "-hf", "test/preset"],
            vec![
                ("Preset", json!(cached(cache, "test/preset", "preset.ini")?)),
                ("ModelPath", json!("")),
                ("ModelHfRepo", json!("")),
            ],
        ),
    ])
}
fn assembly_repos(listings: &Repositories) -> Result<Repositories> {
    let mut repos = BTreeMap::new();
    for (repo, layout) in [
        ("test/main", "flat"),
        ("test/hole", "hole"),
        ("test/quad", "quad"),
        ("test/dflash", "dflash_only"),
        ("test/eagle3", "eagle3_only"),
        ("test/spark", "spark"),
        ("test/pair", "dspark_dflash"),
    ] {
        repos.insert(
            repo.into(),
            listings
                .get(layout)
                .ok_or_else(|| invalid("missing assembly source layout"))?
                .clone(),
        );
    }
    repos.insert("test/small".into(), vec!["draft-model-Q4_K_M.gguf".into()]);
    repos.insert(
        "test/preset".into(),
        vec!["preset.ini".into(), "model-Q8_0.gguf".into()],
    );
    Ok(repos)
}
fn assemblies(
    client: &Client,
    server: &FixtureServer,
    cache: &Path,
    listings: &Repositories,
    checks: &mut Checks,
) -> Result<usize> {
    server.replace(assembly_repos(listings)?)?;
    let cases = assembly_cases(cache)?;
    let initial = checks.source;
    for case in &cases {
        let value = invoke(
            client,
            json!({"Op":"resolution-handler","Arguments":case.arguments,"Example":12,"OfflineAfterInit":true,"Apply":true}),
        )?;
        checks.source(
            value["Parsed"].as_bool() == Some(true),
            format!("assembly CLI parse failed: {:?}", case.arguments),
        );
        for (field, expected) in &case.expected {
            let actual = value["Params"]
                .get(*field)
                .ok_or_else(|| invalid("missing assembled parameter field"))?;
            checks.source(
                actual == expected,
                format!("assembly {field} differs: {:?}", case.arguments),
            );
        }
        checks.guard(
            value["Params"]["Offline"].as_bool() == Some(true),
            "assembly did not apply in original offline mode",
        )?;
    }
    checks.guard(
        cases.len() == 13 && checks.source - initial == 36,
        "original13assembly cases/36predicates changed",
    )?;
    server.replace(BTreeMap::new())?;
    Ok(cases.len())
}
fn cache_refs(cache: &Path, requests: &[Value], checks: &mut Checks) -> Result<Vec<Value>> {
    checks.guard(
        !requests.is_empty(),
        "no live production HTTP requests observed",
    )?;
    let mut repos = std::collections::BTreeSet::new();
    for request in requests {
        let target = text(request, "Target")?;
        let path = target
            .split('?')
            .next()
            .ok_or_else(|| invalid("missing actual HTTP path"))?;
        let repo = path.strip_prefix("/api/models/").and_then(|p| {
            p.strip_suffix("/refs")
                .or_else(|| p.rsplit_once("/tree/").map(|(repo, _)| repo))
        });
        checks.guard(
            request["Method"] == "GET" && request["Status"] == 200 && repo.is_some(),
            "production requested an unexpected fixture route/status",
        )?;
        if let Some(repo) = repo {
            repos.insert(repo.to_owned());
        }
    }
    let mut files = Vec::new();
    for repo in repos {
        let path = cache
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("refs")
            .join("main");
        checks.guard(
            std::fs::metadata(&path)?.len() <= 128,
            "cache ref exceeds bound",
        )?;
        let commit = std::fs::read_to_string(&path)?;
        checks.guard(
            commit == COMMIT,
            "production cache ref did not persist the fixture commit",
        )?;
        files.push(json!({"Repository":repo,"Path":path,"Commit":commit}));
    }
    Ok(files)
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let fixture: Value = serde_json::from_str(INPUTS)?;
    verify(
        fixture["SourceSHA256"] == SOURCE_SHA && fixture["Commit"] == COMMIT,
        "model-resolution provenance changed",
    )?;
    let listings = layouts(&fixture)?;
    verify(
        listings.len() == 10,
        "original listing family inventory changed",
    )?;
    let cache = engine.evidence_path("model-resolution-cache")?;
    let server = FixtureServer::start(COMMIT)?;
    let environment = vec![
        (
            "LLAMA_CACHE".into(),
            cache
                .to_str()
                .ok_or_else(|| invalid("cache path is not UTF-8"))?
                .to_owned(),
        ),
        ("MODEL_ENDPOINT".into(), server.endpoint()),
    ];
    let Some(session) =
        ModelSession::open_with_environment(engine, "model-resolution", &environment)?
    else {
        server.finish()?;
        return Ok(result(
            ID,
            false,
            "production HF resolution/handler/cache API unavailable".into(),
        ));
    };
    let client = session.client();
    let mut checks = Checks::default();
    let outcome = (|| {
        invoke(&client, json!({"Op":"resolution-pause-log"}))?;
        let plans = plans(&client, &server, &fixture, &listings, &mut checks)?;
        let assemblies = assemblies(&client, &server, &cache, &listings, &mut checks)?;
        Ok((plans, assemblies))
    })();
    let finished = session.finish();
    let requests = server.finish();
    let detail = match (outcome, finished, requests) {
        (Ok((plans, assemblies)), Ok(()), Ok(requests)) => {
            let refs = cache_refs(&cache, &requests, &mut checks)?;
            json!({"PlanVariants":plans,"AssemblyCases":assemblies,"HTTPRequests":requests,"CacheRefs":refs,"CacheDirectory":cache})
        }
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
            checks.failed.push(error.to_string());
            json!({"Error":error.to_string()})
        }
    };
    Ok(result(ID,checks.failed.is_empty(),json!({"SourceSHA256":SOURCE_SHA,"SourceChecks":checks.source,"ExpectedSourceChecks":757,"AdditionalGuards":checks.guards,"Observation":detail,"Failures":checks.failed}).to_string()))
}

#[cfg(test)]
mod focused_guards {
    use super::*;
    #[test]
    fn source_inventory_and_mutated_shard_set_are_checked() -> Result<()> {
        let fixture: Value = serde_json::from_str(INPUTS)?;
        let listings = layouts(&fixture)?;
        let cases = fixture["PlanCases"]
            .as_array()
            .ok_or_else(|| invalid("missing cases"))?;
        let mut variants = 0;
        let mut predicates = 0;
        for case in cases {
            let count = if case["OrderDependent"] == true {
                1
            } else {
                listings
                    .get(text(case, "Layout")?)
                    .ok_or_else(|| invalid("missing layout"))?
                    .len()
            };
            variants += count;
            predicates += count * (7 + usize::from(!strings(&case["ModelFiles"])?.is_empty()));
        }
        assert_eq!((variants, predicates), (94, 721));
        let mut checks = Checks::default();
        let case = &cases[0];
        let mut observed = json!({"ModelFiles":[]});
        for key in ["Primary", "Mmproj", "Mtp", "Dflash", "Eagle3", "Dspark"] {
            observed[key] = json!({"Path":case[key]});
        }
        plan_predicates(&observed, case, &mut checks)?;
        assert_eq!(checks.failed.len(), 1);
        let cases = assembly_cases(Path::new("C:\\owned-model-resolution-cache"))?;
        assert_eq!(cases.len(), 13);
        assert_eq!(
            cases.iter().map(|c| 1 + c.expected.len()).sum::<usize>(),
            36
        );
        Ok(())
    }
}

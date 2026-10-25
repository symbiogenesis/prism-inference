// SPDX-License-Identifier: MIT
//! Complete deterministic source geometry identities and input-stream prefix receipts.

use super::super::engine::Engine;
use super::super::{Result, invalid};
use super::graph::Plan;
use super::host::Production;
use super::prepare::{Mode, Prepared, prepare};
use super::program::{self, Geometry, Program, SOURCE_SHA256};
use super::transport;
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn hash(bytes: &[u8]) -> String {
    super::host::encode_hex(&Sha256::digest(bytes))
}
pub(super) fn arguments_hash(geometry: &Geometry) -> Result<String> {
    Ok(hash(&serde_json::to_vec(&geometry.arguments)?))
}
pub(super) fn extent(mode: Mode) -> Result<(&'static Program, &'static [Geometry], &'static str)> {
    let (program, geometries) = program::load().map_err(invalid)?;
    Ok(if mode == Mode::Perf {
        (
            program,
            &geometries.perf,
            "f2dc5bef74e18a5e9740ed77abe9ffd6f5af2ec36c7d00e56d7128bee3205674",
        )
    } else {
        (
            program,
            &geometries.eval,
            "1724d4a88170deb030f676d81f97ce050ab6ee32c37452626fd834ea1c4f7419",
        )
    })
}
pub(super) fn partition(index: usize, count: usize) -> Result<()> {
    if count == 0 || count > 4096 || index >= count {
        return Err(invalid(
            "backend partition requires 0 <= index < count <=4096",
        ));
    }
    Ok(())
}

pub(super) fn receipt(
    host: &Production<'_, '_>,
    prepared: &Prepared,
    geometry: &Geometry,
    index: usize,
) -> Result<Json> {
    let mut contexts = Vec::new();
    for (&context, order) in &host.graph.contexts {
        let tensors = order
            .iter()
            .map(|id| {
                host.graph
                    .tensors
                    .get(id)
                    .ok_or_else(|| invalid("planned tensor metadata missing"))
            })
            .collect::<Result<Vec<_>>>()?;
        contexts.push(json!({"Context":context,"Tensors":tensors.iter().map(|tensor|json!({"Id":tensor.id,"Type":tensor.kind,"NE":tensor.ne,"NB":tensor.nb,"Op":tensor.op,"Flags":tensor.flags,"Name":tensor.name,"Sources":tensor.sources,"OpParamsHex":tensor.op_params_hex,"ViewSource":tensor.view_source,"ViewOffset":tensor.view_offset,"NBytes":tensor.bytes,"NElements":tensor.elements})).collect::<Vec<_>>()}));
    }
    Ok(
        json!({"Index":index,"Class":geometry.class,"ArgumentsSha256":arguments_hash(geometry)?,"ShouldInitialize":prepared.initialize,"CrtDraws":prepared.crt_draws,"Supported":prepared.supported,"SkipReason":prepared.skip,"Operation":prepared.operation,"Variables":prepared.variables,"ReferenceAvailable":prepared.reference_available,"MetadataSha256":hash(&serde_json::to_vec(&contexts)?),"SourceAssertions":host.assertions}),
    )
}

pub(super) fn plan(
    engine: &Engine,
    index: usize,
    count: usize,
    mode: Mode,
    filter: Option<&str>,
) -> Result<Json> {
    partition(index, count)?;
    let (program, geometries, geometry_hash) = extent(mode)?;
    let threads = std::thread::available_parallelism()?.get();
    let mut items = Vec::new();
    let partition = transport::Partition::open(engine)?;
    for (case_index, geometry) in geometries
        .iter()
        .enumerate()
        .filter(|(case_index, _)| case_index % count == index)
    {
        let directory =
            engine.evidence_path(&format!("backend-{}-plan-{case_index:05}", mode.name()))?;
        let observed = partition.with_session(&directory, |session| {
            let mut host = Production::new(
                Plan::new(session),
                program,
                case_index as u64,
                threads,
                mode.code(),
            );
            let prepared = prepare(&mut host, geometry, mode, filter)?;
            receipt(&host, &prepared, geometry, case_index)
        })?;
        let Some(observed) = observed else {
            return Ok(
                json!({"SchemaVersion":1,"SourceId":"test-backend-ops","SourceSha256":SOURCE_SHA256,"GeometrySha256":geometry_hash,"Mode":mode.name(),"CpuSelection":"explicit CPU","Seed":1,"Partition":{"Index":index,"Count":count},"Status":"unsupported","Reason":"unsupported production API: generic backend graph, support and metadata observations","ExpectedGeometries":geometries.len(),"Items":items}),
            );
        };
        std::fs::write(
            directory.join("case-receipt.json"),
            serde_json::to_vec_pretty(&observed)?,
        )?;
        super::driver::cleanup(&directory)?;
        items.push(observed);
    }
    partition.finish()?;
    Ok(
        json!({"SchemaVersion":1,"SourceId":"test-backend-ops","SourceSha256":SOURCE_SHA256,"GeometrySha256":geometry_hash,"Mode":mode.name(),"CpuSelection":"explicit CPU","Seed":1,"Partition":{"Index":index,"Count":count},"Status":"planned","ExpectedGeometries":geometries.len(),"Filter":filter,"Threads":threads,"Items":items}),
    )
}

#[allow(
    clippy::too_many_lines,
    reason = "All partition, ordered geometry and CRT-prefix completeness checks remain in one aggregation transaction."
)]
pub(super) fn aggregate(plans: &[Json]) -> Result<Json> {
    let first = plans
        .first()
        .ok_or_else(|| invalid("backend planner aggregation requires all partitions"))?;
    let mode = Mode::parse(
        first["Mode"]
            .as_str()
            .ok_or_else(|| invalid("backend plan mode missing"))?,
    )?;
    let (_, geometries, geometry_hash) = extent(mode)?;
    let count = usize::try_from(
        first["Partition"]["Count"]
            .as_u64()
            .ok_or_else(|| invalid("backend plan partition count missing"))?,
    )?;
    if plans.len() != count {
        return Err(invalid("backend planner partition count incomplete"));
    }
    let mut partitions = BTreeSet::new();
    let mut items = BTreeMap::new();
    for plan in plans {
        if plan["SchemaVersion"] != 1
            || plan["SourceId"] != "test-backend-ops"
            || plan["SourceSha256"] != SOURCE_SHA256
            || plan["GeometrySha256"] != geometry_hash
            || plan["Mode"] != mode.name()
            || plan["CpuSelection"] != "explicit CPU"
            || plan["Seed"] != 1
            || plan["Status"] != "planned"
            || plan["ExpectedGeometries"] != geometries.len()
            || plan["Filter"] != first["Filter"]
            || plan["Threads"] != first["Threads"]
            || plan["Partition"]["Count"] != count
        {
            return Err(invalid(
                "backend planner provenance, mode, CPU selection or completeness differs",
            ));
        }
        let index = usize::try_from(
            plan["Partition"]["Index"]
                .as_u64()
                .ok_or_else(|| invalid("backend plan partition index missing"))?,
        )?;
        partition(index, count)?;
        if !partitions.insert(index) {
            return Err(invalid("duplicate backend planner partition"));
        }
        let rows = plan["Items"]
            .as_array()
            .ok_or_else(|| invalid("backend plan rows missing"))?;
        let expected = geometries
            .iter()
            .enumerate()
            .filter(|(case_index, _)| case_index % count == index)
            .map(|(case_index, _)| case_index)
            .collect::<Vec<_>>();
        if rows.len() != expected.len() {
            return Err(invalid("backend planner partition geometry extent differs"));
        }
        for (row, &expected_index) in rows.iter().zip(&expected) {
            let case_index = usize::try_from(
                row["Index"]
                    .as_u64()
                    .ok_or_else(|| invalid("backend planned geometry index missing"))?,
            )?;
            if case_index != expected_index
                || row["Class"] != geometries[case_index].class
                || row["ArgumentsSha256"] != arguments_hash(&geometries[case_index])?
                || row["ShouldInitialize"].as_bool().is_none()
                || row["CrtDraws"].as_u64().is_none()
                || row["Supported"].as_bool().is_none()
            {
                return Err(invalid(
                    "backend planned geometry source identity/order differs",
                ));
            }
            if items.insert(case_index, row.clone()).is_some() {
                return Err(invalid("duplicate backend planned geometry"));
            }
        }
    }
    if items.len() != geometries.len() {
        return Err(invalid("backend planner complete source extent missing"));
    }
    let mut offset = 0_u64;
    let mut ordered = Vec::with_capacity(items.len());
    for (case_index, mut row) in items {
        if case_index != ordered.len() {
            return Err(invalid("backend planner source order gap"));
        }
        row["CrtOffset"] = json!(offset);
        offset = offset
            .checked_add(
                row["CrtDraws"]
                    .as_u64()
                    .ok_or_else(|| invalid("backend planner CRT count missing"))?,
            )
            .ok_or_else(|| invalid("backend planner CRT prefix overflow"))?;
        ordered.push(row);
    }
    let mut combined = json!({"SchemaVersion":1,"SourceId":"test-backend-ops","SourceSha256":SOURCE_SHA256,"GeometrySha256":geometry_hash,"Mode":mode.name(),"CpuSelection":"explicit CPU","Seed":1,"Status":"planned-complete","ExpectedGeometries":geometries.len(),"Filter":first["Filter"],"Threads":first["Threads"],"Items":ordered,"TotalCrtDraws":offset});
    combined["PlanSha256"] = json!(hash(&serde_json::to_vec(&combined)?));
    Ok(combined)
}

pub(super) fn validate(plan: &Json, mode: Mode) -> Result<&[Json]> {
    let (_, geometries, geometry_hash) = extent(mode)?;
    let mut unsigned = plan.clone();
    let claimed = unsigned
        .as_object_mut()
        .ok_or_else(|| invalid("backend aggregate plan is not an object"))?
        .remove("PlanSha256")
        .ok_or_else(|| invalid("backend aggregate plan digest missing"))?;
    if claimed != hash(&serde_json::to_vec(&unsigned)?)
        || plan["SchemaVersion"] != 1
        || plan["SourceId"] != "test-backend-ops"
        || plan["SourceSha256"] != SOURCE_SHA256
        || plan["GeometrySha256"] != geometry_hash
        || plan["Mode"] != mode.name()
        || plan["CpuSelection"] != "explicit CPU"
        || plan["Seed"] != 1
        || plan["Status"] != "planned-complete"
        || plan["ExpectedGeometries"] != geometries.len()
    {
        return Err(invalid(
            "backend aggregate plan provenance/digest/completeness differs",
        ));
    }
    let items = plan["Items"]
        .as_array()
        .filter(|items| items.len() == geometries.len())
        .ok_or_else(|| invalid("backend aggregate plan complete geometry extent missing"))?;
    let mut offset = 0_u64;
    for (index, row) in items.iter().enumerate() {
        if row["Index"] != index
            || row["Class"] != geometries[index].class
            || row["ArgumentsSha256"] != arguments_hash(&geometries[index])?
            || row["CrtOffset"] != offset
        {
            return Err(invalid(
                "backend aggregate geometry or CRT prefix identity differs",
            ));
        }
        offset = offset
            .checked_add(
                row["CrtDraws"]
                    .as_u64()
                    .ok_or_else(|| invalid("backend aggregate CRT count missing"))?,
            )
            .ok_or_else(|| invalid("backend aggregate CRT prefix overflow"))?;
    }
    if plan["TotalCrtDraws"] != offset {
        return Err(invalid("backend aggregate total CRT draws differ"));
    }
    Ok(items)
}

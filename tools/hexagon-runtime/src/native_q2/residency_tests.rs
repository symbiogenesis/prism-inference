//! Shared-source geometry and stable-index controls, without native execution.

use super::*;

#[test]
fn resident_cache_index_separates_shared_storage_shapes_and_keeps_handles_stable() -> Result<()> {
    let mut cache = ResidentCache::new(4096)?;
    let source: Arc<[u8]> = Arc::from(vec![0x55; 128 * 34]);
    let mut tensors = Vec::new();
    for (columns, rows) in [(512, 32), (256, 64), (128, 128)] {
        tensors.push(Tensor {
            name: "shared-storage-control".into(),
            kind: 142,
            dimensions: vec![columns, rows],
            data: Arc::clone(&source).into(),
        });
    }
    let capacities = (cache.projections.capacity(), cache.by_storage.capacity());
    for (index, tensor) in tensors.iter().enumerate() {
        assert_eq!(cache.commit(tensor, index * 7, index == 0), index);
        for (prior, selected) in tensors[..=index].iter().enumerate() {
            assert_eq!(cache.find(selected), Some(prior));
            assert_eq!(cache.projections[prior].first_handle, prior * 7);
            assert_eq!(cache.projections[prior].evaluated, prior == 0);
        }
    }
    for tensor in tensors {
        let mut changed = tensor.clone();
        changed.data = Arc::<[u8]>::from(tensor.data.to_vec()).into();
        assert_eq!(cache.find(&changed), None);
        changed = tensor;
        changed.kind = 143;
        assert_eq!(cache.find(&changed), None);
        changed.dimensions.clear();
        assert_eq!(cache.find(&changed), None);
    }
    assert_eq!(
        capacities,
        (cache.projections.capacity(), cache.by_storage.capacity())
    );
    assert!(cache.statistics.metadata_payload_bytes <= cache.reservation);
    Ok(())
}

#[test]
fn resident_cache_index_native_metadata_reservation_covers_every_vector() -> Result<()> {
    let bytes =
        size_of::<ResidentProjection>() + size_of::<NativeQ2WeightHandle>() + size_of::<usize>();
    assert!(ResidentCache::new(bytes - 1).is_err());
    let mut cache = ResidentCache::new(bytes)?;
    assert_eq!(cache.statistics.metadata_payload_bytes, bytes);
    let tensor = Tensor {
        name: "one-slot".into(),
        kind: 142,
        dimensions: vec![128, 32],
        data: Arc::<[u8]>::from(vec![0x55; 32 * 34]).into(),
    };
    assert!(cache.has_metadata_capacity(1));
    cache.commit(&tensor, 0, false);
    assert!(!cache.has_metadata_capacity(1));
    Ok(())
}

#[test]
fn resident_cache_index_admits_ptq1_and_separates_from_pq2_alias() -> Result<()> {
    let mut cache = ResidentCache::new(4096)?;
    let shared: Arc<[u8]> = Arc::from(vec![0x55; 128 * 34]);
    let ptq1 = Tensor {
        name: "shared-ptq1".into(),
        kind: 143,
        dimensions: vec![128, 32],
        data: Arc::clone(&shared).into(),
    };
    let pq2_alias = Tensor {
        name: "shared-pq2".into(),
        kind: 142,
        dimensions: vec![128, 32],
        data: shared.into(),
    };
    assert_eq!(cache.commit(&ptq1, 3, true), 0);
    assert_eq!(cache.find(&ptq1), Some(0));
    assert_eq!(cache.find(&pq2_alias), None);
    assert_eq!(cache.commit(&pq2_alias, 9, false), 1);
    assert_eq!(cache.find(&ptq1), Some(0));
    assert_eq!(cache.find(&pq2_alias), Some(1));
    Ok(())
}

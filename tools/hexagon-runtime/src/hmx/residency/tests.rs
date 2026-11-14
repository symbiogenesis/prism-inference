//! Independent identity and capacity controls for the resident lookup index.

use super::*;

fn tensor(columns: usize, rows: usize, data: Arc<[u8]>) -> Tensor {
    Tensor {
        name: "resident-index-control".into(),
        dimensions: vec![columns, rows],
        kind: 142,
        data,
    }
}

#[test]
fn resident_cache_index_matches_linear_identity_and_preserves_append_order() -> Result<()> {
    let mut cache = Cache::new(32 * 1024)?;
    let shared: Arc<[u8]> = Arc::from(vec![0x55; 128 * 34]);
    let mut tensors = vec![
        tensor(512, 32, Arc::clone(&shared)),
        tensor(256, 64, Arc::clone(&shared)),
        tensor(128, 128, shared),
    ];
    for _ in 0..165 {
        tensors.push(tensor(128, 128, Arc::from(vec![0x55; 128 * 34])));
    }
    // Insert in reverse search-key order so every insertion exercises an
    // earlier index position, independently of allocator address order.
    tensors.sort_by_key(|tensor| {
        std::cmp::Reverse((
            tensor.data.as_ptr().addr(),
            tensor.dimensions[0],
            tensor.dimensions[1],
        ))
    });
    let capacities = (cache.projections.capacity(), cache.by_storage.capacity());
    for (expected, selected) in tensors.iter().enumerate() {
        assert!(cache.has_metadata_capacity(1));
        assert_eq!(cache.commit(selected, expected * 3), expected);
        for control in &tensors[..=expected] {
            let oracle = cache.projections.iter().position(|projection| {
                Arc::ptr_eq(&projection.source, &control.data)
                    && control.dimensions == [projection.columns, projection.rows]
            });
            assert_eq!(cache.find(control), oracle);
        }
    }
    for (index, selected) in tensors.iter().enumerate() {
        assert_eq!(cache.find(selected), Some(index));
        assert_eq!(cache.projections[index].first_handle, index * 3);
        let mut renamed = selected.clone();
        renamed.name.push_str("-renamed");
        assert_eq!(cache.find(&renamed), Some(index));
        let mut copied = selected.clone();
        copied.data = Arc::from(selected.data.to_vec());
        assert_eq!(cache.find(&copied), None);
        let mut malformed = selected.clone();
        malformed.kind = 143;
        assert_eq!(cache.find(&malformed), None);
        malformed.dimensions.clear();
        assert_eq!(cache.find(&malformed), None);
    }
    assert_eq!(
        capacities,
        (cache.projections.capacity(), cache.by_storage.capacity())
    );
    assert!(cache.statistics.metadata_payload_bytes <= cache.reservation);
    Ok(())
}

#[test]
fn resident_cache_index_storage_is_charged_and_cannot_grow_past_reservation() -> Result<()> {
    let bytes = size_of::<Projection>() + size_of::<HmxQ2WeightHandle>() + size_of::<usize>();
    assert!(Cache::new(bytes - 1).is_err());
    let mut cache = Cache::new(bytes)?;
    assert_eq!(cache.statistics.metadata_payload_bytes, bytes);
    assert!(cache.has_metadata_capacity(1));
    cache.commit(&tensor(128, 32, Arc::from(vec![0x55; 32 * 34])), 0);
    assert!(!cache.has_metadata_capacity(1));
    Ok(())
}

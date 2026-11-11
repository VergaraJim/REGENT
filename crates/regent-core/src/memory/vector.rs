//! Vector maths and the exact brute-force cosine index.

use std::collections::HashMap;

/// L2-normalises `v` in place.
///
/// Returns `false` and leaves `v` unchanged when its norm is zero or not
/// finite.
pub fn normalize(v: &mut [f32]) -> bool {
    let sum: f32 = v.iter().map(|value| value * value).sum();
    let norm = sum.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return false;
    }
    for value in v.iter_mut() {
        *value /= norm;
    }
    true
}

/// The dot product of two equal-length slices.
///
/// On normalised vectors this is their cosine similarity.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One recalled message and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// The message that matched.
    pub message_id: i64,
    /// The cosine score of its best chunk against the query.
    pub score: f32,
}

/// An exact brute-force cosine index over normalised vectors.
///
/// This is the spec's exact fallback (§5.6): it compares every stored chunk
/// with the query, so it gives the same ranking as an accelerated index, just
/// more slowly. A message with several chunks scores as its best chunk.
#[derive(Debug, Clone, Default)]
pub struct VectorIndex {
    dimension: Option<usize>,
    chunks: Vec<(i64, Vec<f32>)>,
}

impl VectorIndex {
    /// An empty index; the first insert fixes its dimension.
    pub fn new() -> VectorIndex {
        VectorIndex::default()
    }

    /// The embedding dimension, once something has been inserted.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// How many chunks are stored.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Whether nothing has been inserted yet.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Adds one chunk of `message_id`, normalising it.
    ///
    /// The first insert fixes the dimension. A vector of another dimension is
    /// refused and returns `false`.
    pub fn insert(&mut self, message_id: i64, mut vector: Vec<f32>) -> bool {
        match self.dimension {
            None => self.dimension = Some(vector.len()),
            Some(dimension) if vector.len() != dimension => return false,
            Some(_) => {}
        }
        normalize(&mut vector);
        self.chunks.push((message_id, vector));
        true
    }

    /// The `limit` best messages for `query`, excluding ids `exclude` accepts.
    ///
    /// Ties are broken by the newer (higher) id. A query of the wrong
    /// dimension returns an empty result.
    pub fn search(&self, query: &[f32], limit: usize, exclude: &dyn Fn(i64) -> bool) -> Vec<Hit> {
        let Some(dimension) = self.dimension else {
            return Vec::new();
        };
        if query.len() != dimension {
            return Vec::new();
        }

        let mut query = query.to_vec();
        normalize(&mut query);

        let mut best: HashMap<i64, f32> = HashMap::new();
        for (message_id, vector) in &self.chunks {
            if exclude(*message_id) {
                continue;
            }
            let score = dot(&query, vector);
            best.entry(*message_id)
                .and_modify(|current| {
                    if score > *current {
                        *current = score;
                    }
                })
                .or_insert(score);
        }

        let mut hits: Vec<Hit> = best
            .into_iter()
            .map(|(message_id, score)| Hit { message_id, score })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.message_id.cmp(&a.message_id))
        });
        hits.truncate(limit);
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none(_: i64) -> bool {
        false
    }

    #[test]
    fn normalize_gives_a_unit_vector() {
        let mut v = [3.0, 4.0];
        assert!(normalize(&mut v));
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        assert!((dot(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_refuses_a_zero_vector() {
        let mut v = [0.0, 0.0, 0.0];
        assert!(!normalize(&mut v));
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_message_scores_as_its_best_chunk() {
        let mut index = VectorIndex::new();
        assert!(index.is_empty());
        assert!(index.insert(1, vec![1.0, 0.0]));
        assert!(index.insert(1, vec![0.0, 1.0]));
        assert!(index.insert(2, vec![0.6, 0.8]));
        assert!(index.insert(2, vec![0.8, 0.6]));
        assert_eq!(index.dimension(), Some(2));
        assert_eq!(index.chunk_count(), 4);

        let hits = index.search(&[1.0, 0.0], 10, &none);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].message_id, 1);
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        assert_eq!(hits[1].message_id, 2);
        assert!((hits[1].score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn excluded_ids_are_skipped() {
        let mut index = VectorIndex::new();
        assert!(index.insert(1, vec![1.0, 0.0]));
        assert!(index.insert(2, vec![0.9, 0.1]));

        let hits = index.search(&[1.0, 0.0], 10, &|id| id == 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].message_id, 2);
    }

    #[test]
    fn the_limit_truncates_the_hits() {
        let mut index = VectorIndex::new();
        assert!(index.insert(1, vec![1.0, 0.0]));
        assert!(index.insert(2, vec![0.9, 0.1]));
        assert!(index.insert(3, vec![0.8, 0.2]));

        let hits = index.search(&[1.0, 0.0], 1, &none);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].message_id, 1);
    }

    #[test]
    fn ties_go_to_the_newer_message() {
        let mut index = VectorIndex::new();
        assert!(index.insert(1, vec![1.0, 0.0]));
        assert!(index.insert(7, vec![1.0, 0.0]));

        let hits = index.search(&[1.0, 0.0], 10, &none);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].message_id, 7);
        assert_eq!(hits[1].message_id, 1);
    }

    #[test]
    fn a_wrong_query_dimension_matches_nothing() {
        let mut index = VectorIndex::new();
        assert!(index.insert(1, vec![1.0, 0.0]));
        assert!(index.search(&[1.0, 0.0, 0.0], 10, &none).is_empty());
        assert!(VectorIndex::new().search(&[1.0], 10, &none).is_empty());
    }

    #[test]
    fn a_different_dimension_is_refused() {
        let mut index = VectorIndex::new();
        assert!(index.insert(1, vec![1.0, 0.0]));
        assert!(!index.insert(2, vec![1.0, 0.0, 0.0]));
        assert_eq!(index.chunk_count(), 1);
    }
}

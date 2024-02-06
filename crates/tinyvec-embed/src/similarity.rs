use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Distance {
    Cosine,
}

impl Distance {
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Distance::Cosine => {
                let magnitude_a = a.iter().fold(0.0, |acc, &val| val.mul_add(val, acc));
                let magnitude_b = b.iter().fold(0.0, |acc, &val| val.mul_add(val, acc));
                dot_product(a, b) / (magnitude_a * magnitude_b).sqrt()
            }
        }
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |acc, (x, y)| acc + x * y)
}

pub struct ScoreIndex {
    pub score: f32,
    pub index: usize,
}

impl PartialEq for ScoreIndex {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl Eq for ScoreIndex {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for ScoreIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // The comparison is intentionally reversed here to make the heap a min-heap
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for ScoreIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

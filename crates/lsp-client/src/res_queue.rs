use std::collections::HashMap;

use serde::Serialize;

use crate::msg::{Request, RequestId};

/// Manages the set of pending responses
#[derive(Debug)]
pub struct ResQueue<O> {
    pub outgoing: Outgoing<O>,
}

impl<O> Default for ResQueue<O> {
    fn default() -> ResQueue<O> {
        ResQueue {
            outgoing: Outgoing {
                next_id: 0,
                pending: HashMap::default(),
            },
        }
    }
}

#[derive(Debug)]
pub struct Outgoing<O> {
    next_id: i32,
    pending: HashMap<RequestId, O>,
}

impl<O> Outgoing<O> {
    pub fn register<P: Serialize>(&mut self, method: String, params: P, data: O) -> Request {
        let id = RequestId::from(self.next_id);
        self.pending.insert(id.clone(), data);
        self.next_id += 1;
        Request::new(id, method, params)
    }

    pub fn complete(&mut self, id: RequestId) -> Option<O> {
        self.pending.remove(&id)
    }
}

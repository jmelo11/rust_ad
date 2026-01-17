use std::{fmt, ptr::NonNull};

#[derive(Clone)]
pub struct TapeNode {
    pub childs: Vec<NonNull<TapeNode>>,
    pub derivs: Vec<f64>,
    pub adj: f64,
}

impl fmt::Debug for TapeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TapeNode {{ addr: {:?}, childs: {:?}, derivs: {:?}, adj: {} }}",
            self as *const Self as *const (), self.childs, self.derivs, self.adj
        )
    }
}

impl Default for TapeNode {
    fn default() -> Self {
        Self {
            childs: Vec::new(),
            derivs: Vec::new(),
            adj: 0.0,
        }
    }
}

impl TapeNode {
    #[inline(always)]
    pub fn propagate_into(&self) {
        debug_assert_eq!(self.childs.len(), self.derivs.len());
        let a = self.adj;
        for (&child, &d) in self.childs.iter().zip(&self.derivs) {
            unsafe { (*child.as_ptr()).adj += a * d };
        }
    }
}

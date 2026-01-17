use bumpalo::Bump;
use std::{cell::RefCell, ptr::NonNull};

use crate::errors::{ADError, Result};
use crate::prelude::TapeNode;

pub struct Tape {
    bump: Bump,
    book: Vec<NonNull<TapeNode>>,
    mark: usize,
    pub active: bool,
}

impl Tape {
    #[inline(always)]
    fn push(&mut self, n: TapeNode) -> NonNull<TapeNode> {
        let ptr = NonNull::from(self.bump.alloc(n));
        self.book.push(ptr);
        ptr
    }

    #[inline(always)]
    pub fn reset_adjoints() {
        TAPE.with(|tc| {
            for &ptr in &tc.borrow().book {
                unsafe { (*ptr.as_ptr()).adj = 0.0 };
            }
        });
    }

    pub fn debug_print() {
        TAPE.with(|tc| {
            let tape = tc.borrow();
            for (i, &ptr) in tape.book.iter().enumerate() {
                let node = unsafe { ptr.as_ref() };
                println!("{}: {:?}", i, node);
            }
        });
    }

    pub fn mark(&self) -> usize {
        self.mark
    }

    #[inline(always)]
    fn index_of(&self, p: NonNull<TapeNode>) -> Option<usize> {
        self.book.iter().position(|&q| q == p)
    }
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            bump: Bump::new(),
            book: Vec::new(),
            mark: 0,
            active: false,
        }
    }

    #[inline]
    pub fn new_leaf(&mut self) -> NonNull<TapeNode> {
        self.push(TapeNode::default())
    }

    #[inline]
    pub fn record(&mut self, n: TapeNode) -> Option<NonNull<TapeNode>> {
        self.active.then(|| self.push(n))
    }

    pub fn node(&self, p: NonNull<TapeNode>) -> Option<&TapeNode> {
        self.index_of(p).map(|i| unsafe { self.book[i].as_ref() })
    }
    pub fn mut_node(&mut self, p: NonNull<TapeNode>) -> Option<&mut TapeNode> {
        self.index_of(p).map(|i| unsafe { self.book[i].as_mut() })
    }

    pub fn propagate_from(&mut self, root: NonNull<TapeNode>) -> Result<()> {
        let start = self
            .index_of(root)
            .ok_or(ADError::NodeNotIndexedInTapeErr)?;
        for i in (0..=start).rev() {
            let node = unsafe { self.book[i].as_ref().clone() };
            node.propagate_into();
        }
        Ok(())
    }

    pub fn propagate_mark_to_start(&mut self) {
        let end = self.mark.saturating_sub(1);
        for i in (0..=end).rev() {
            let node = unsafe { self.book[i].as_ref().clone() };
            node.propagate_into();
        }
    }

    pub fn propagate_to_mark(&mut self) {
        let end = self.mark.saturating_sub(1);
        for i in (0..=end).rev() {
            let node = unsafe { self.book[i].as_ref().clone() };
            node.propagate_into();
        }
    }

    pub fn start_recording() {
        TAPE.with(|tc| {
            let mut t = tc.borrow_mut();
            t.bump.reset();
            t.book.clear();
            t.mark = 0;
            t.active = true;
        });
    }

    pub fn stop_recording() {
        TAPE.with(|tc| tc.borrow_mut().active = false);
    }

    #[inline]
    pub fn is_active() -> bool {
        TAPE.with(|tc| tc.borrow().active)
    }

    pub fn set_mark() {
        TAPE.with(|tc| {
            let len = tc.borrow().book.len();
            tc.borrow_mut().mark = len;
        });
    }

    pub fn rewind_to_mark() {
        TAPE.with(|tc| {
            let mark = tc.borrow().mark;
            tc.borrow_mut().book.truncate(mark);
        });
    }

    pub fn rewind_to_init() {
        TAPE.with(|tc| {
            let mut t = tc.borrow_mut();
            t.bump.reset();
            t.book.clear();
            t.mark = 0;
        });
    }
}

thread_local! {
    pub static TAPE: RefCell<Tape> = RefCell::new(Tape {
        bump:   Bump::new(),
        book:   Vec::new(),
        mark:   0,
        active: false,
    });
}

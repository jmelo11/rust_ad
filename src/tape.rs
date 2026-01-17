//! Tape storage and propagation utilities.

use bumpalo::Bump;
use std::{cell::RefCell, ptr::NonNull};

use crate::errors::{ADError, Result};
use crate::prelude::TapeNode;

/// A tape holding all recorded nodes for reverse-mode differentiation.
pub struct Tape {
    bump: Bump,
    book: Vec<NonNull<TapeNode>>,
    mark: usize,
    /// Whether the tape should record new nodes.
    pub active: bool,
}

impl Tape {
    #[inline(always)]
    /// Allocates a node in the bump arena and records it in the tape book.
    fn push(&mut self, n: TapeNode) -> NonNull<TapeNode> {
        let ptr = NonNull::from(self.bump.alloc(n));
        self.book.push(ptr);
        ptr
    }

    #[inline(always)]
    /// Resets all adjoints on the current thread's tape.
    pub fn reset_adjoints() {
        TAPE.with(|tc| {
            for &ptr in &tc.borrow().book {
                unsafe { (*ptr.as_ptr()).adj = 0.0 };
            }
        });
    }

    /// Prints the tape contents to stdout for debugging.
    pub fn debug_print() {
        TAPE.with(|tc| {
            let tape = tc.borrow();
            for (i, &ptr) in tape.book.iter().enumerate() {
                let node = unsafe { ptr.as_ref() };
                println!("{}: {:?}", i, node);
            }
        });
    }

    /// Returns the current mark index for this tape.
    pub fn mark(&self) -> usize {
        self.mark
    }

    #[inline(always)]
    /// Returns the index of a node in the tape book, if it exists.
    ///
    /// This is a linear scan; the cost grows with the number of recorded nodes.
    fn index_of(&self, p: NonNull<TapeNode>) -> Option<usize> {
        self.book.iter().position(|&q| q == p)
    }
}

impl Tape {
    /// Creates an empty tape with recording disabled.
    pub fn new() -> Self {
        Tape {
            bump: Bump::new(),
            book: Vec::new(),
            mark: 0,
            active: false,
        }
    }

    #[inline]
    /// Allocates and records a leaf node.
    pub fn new_leaf(&mut self) -> NonNull<TapeNode> {
        self.push(TapeNode::default())
    }

    #[inline]
    /// Records a node if recording is active, returning its pointer.
    pub fn record(&mut self, n: TapeNode) -> Option<NonNull<TapeNode>> {
        self.active.then(|| self.push(n))
    }

    /// Retrieves an immutable reference to a node by pointer.
    pub fn node(&self, p: NonNull<TapeNode>) -> Option<&TapeNode> {
        self.index_of(p).map(|i| unsafe { self.book[i].as_ref() })
    }
    /// Retrieves a mutable reference to a node by pointer.
    pub fn mut_node(&mut self, p: NonNull<TapeNode>) -> Option<&mut TapeNode> {
        self.index_of(p).map(|i| unsafe { self.book[i].as_mut() })
    }

    /// Propagates adjoints from the given root node back to the start of the tape.
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

    /// Propagates adjoints from the current mark back to the start.
    pub fn propagate_mark_to_start(&mut self) {
        let end = self.mark.saturating_sub(1);
        for i in (0..=end).rev() {
            let node = unsafe { self.book[i].as_ref().clone() };
            node.propagate_into();
        }
    }

    /// Propagates adjoints from the end of the tape down to the current mark.
    pub fn propagate_to_mark(&mut self) {
        let start = self.mark;
        let end = self.book.len().saturating_sub(1);
        if start > end {
            return;
        }
        for i in (start..=end).rev() {
            let node = unsafe { self.book[i].as_ref().clone() };
            node.propagate_into();
        }
    }

    /// Clears the tape and begins recording nodes in the thread-local tape.
    pub fn start_recording() {
        TAPE.with(|tc| {
            let mut t = tc.borrow_mut();
            t.bump.reset();
            t.book.clear();
            t.mark = 0;
            t.active = true;
        });
    }

    /// Stops recording nodes on the thread-local tape.
    pub fn stop_recording() {
        TAPE.with(|tc| tc.borrow_mut().active = false);
    }

    #[inline]
    /// Returns whether the thread-local tape is active.
    pub fn is_active() -> bool {
        TAPE.with(|tc| tc.borrow().active)
    }

    /// Sets the current mark to the end of the tape.
    pub fn set_mark() {
        TAPE.with(|tc| {
            let len = tc.borrow().book.len();
            tc.borrow_mut().mark = len;
        });
    }

    /// Truncates the tape back to the current mark.
    pub fn rewind_to_mark() {
        TAPE.with(|tc| {
            let mark = tc.borrow().mark;
            tc.borrow_mut().book.truncate(mark);
        });
    }

    /// Clears the tape and resets the mark without changing active state.
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
    /// Thread-local tape used by default by `ADNumber`.
    pub static TAPE: RefCell<Tape> = RefCell::new(Tape {
        bump:   Bump::new(),
        book:   Vec::new(),
        mark:   0,
        active: false,
    });
}

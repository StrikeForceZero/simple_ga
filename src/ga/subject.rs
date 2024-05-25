use std::fmt::{Debug, Formatter};

#[derive(Clone)]
pub struct Subject<T: Clone> {
    pub generation_born: u32,
    pub data: T,
}

impl<T: Debug + Clone> Debug for Subject<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subject")
            .field("generation_born", &self.generation_born)
            .field("data", &self.data)
            .finish()
    }
}

#[cfg(not(feature = "parallel"))]
pub trait GaSubject {}

#[cfg(feature = "parallel")]
pub trait GaSubject: Send + Sync {}

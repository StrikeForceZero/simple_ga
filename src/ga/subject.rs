#[derive(Clone)]
pub struct Subject<T: Clone> {
    pub generation_born: u32,
    pub data: T,
}

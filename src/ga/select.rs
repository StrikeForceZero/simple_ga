use std::collections::HashSet;

use rand::prelude::ThreadRng;

use crate::util::{Bias, random_index_bias};

pub trait SelectRandomOwned<Iter>
where
    Iter: IntoIterator + ExactSizeIterator,
{
    type Output;
    fn select_random_owned(self, rng: &mut ThreadRng, items: Iter) -> Self::Output;
}

pub trait SelectRandom<T> {
    type Output<'a>
    where
        T: 'a;
    fn select_random<'a>(self, rng: &mut ThreadRng, items: &'a [T]) -> Self::Output<'a>;
}

pub trait SelectRandomMut<'a, T>
where
    T: 'a,
{
    type Output;
    fn select_random_mut(self, rng: &mut ThreadRng, items: &'a mut [T]) -> Self::Output;
}

#[derive(Debug, Copy, Clone)]
pub struct SelectRandomWithBias {
    bias: Bias,
}

impl SelectRandomWithBias {
    pub fn new(bias: Bias) -> Self {
        Self { bias }
    }
    pub fn bias(&self) -> &Bias {
        &self.bias
    }
}

impl<Iter> SelectRandomOwned<Iter> for SelectRandomWithBias
where
    Iter: IntoIterator + ExactSizeIterator,
{
    type Output = Option<<Iter as IntoIterator>::Item>;
    fn select_random_owned(self, rng: &mut ThreadRng, items: Iter) -> Self::Output {
        let index = random_index_bias(rng, items.len(), self.bias);
        items.into_iter().nth(index)
    }
}

impl<T> SelectRandom<T> for SelectRandomWithBias {
    type Output<'a> = Option<&'a T> where T: 'a;
    fn select_random<'a>(self, rng: &mut ThreadRng, items: &'a [T]) -> Self::Output<'a> {
        items.get(random_index_bias(rng, items.len(), self.bias))
    }
}

impl<'a, T> SelectRandomMut<'a, &'a mut T> for SelectRandomWithBias
where
    T: 'a,
{
    type Output = Option<&'a mut &'a mut T>;
    fn select_random_mut(self, rng: &mut ThreadRng, items: &'a mut [&'a mut T]) -> Self::Output {
        items.get_mut(random_index_bias(rng, items.len(), self.bias))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SelectRandomManyWithBias {
    amount: usize,
    bias: Bias,
}

impl SelectRandomManyWithBias {
    pub fn new(amount: usize, bias: Bias) -> Self {
        Self { amount, bias }
    }
    pub fn amount(&self) -> &usize {
        &self.amount
    }
    pub fn bias(&self) -> &Bias {
        &self.bias
    }
    fn select_random_indexes(&self, rng: &mut ThreadRng, len: usize) -> HashSet<usize> {
        let max_amount = self.amount.min(len);
        // not enough items just return the original slice as a new vec
        if max_amount >= len {
            (0..len).collect()
        } else {
            let mut selected_indexes = HashSet::new();
            while selected_indexes.len() < max_amount {
                selected_indexes.insert(random_index_bias(rng, len, self.bias));
            }
            selected_indexes.into_iter().collect()
        }
    }
    fn select_random<'a, T>(self, rng: &mut ThreadRng, items: &'a [T]) -> Vec<&'a T> {
        self.select_random_indexes(rng, items.len())
            .into_iter()
            .filter_map(|ix| items.get(ix))
            .collect()
    }
    fn select_random_mut<'a, T>(
        self,
        rng: &mut ThreadRng,
        items: &'a mut [&'a mut T],
    ) -> Vec<&'a mut &'a mut T> {
        let mut selected_indexes = self.select_random_indexes(rng, items.len());
        items
            .iter_mut()
            .enumerate()
            .filter_map(|(ix, v)| {
                if selected_indexes.remove(&ix) {
                    Some(v)
                } else {
                    None
                }
            })
            .collect()
    }
    fn select_random_owned<'a, Iter, Item>(self, rng: &mut ThreadRng, items: Iter) -> Vec<Item>
    where
        Iter: IntoIterator<Item = Item> + ExactSizeIterator,
    {
        let mut selected_indexes = self.select_random_indexes(rng, items.len());
        items
            .into_iter()
            .enumerate()
            .filter_map(|(ix, v)| {
                if selected_indexes.remove(&ix) {
                    Some(v)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl<Iter> SelectRandomOwned<Iter> for SelectRandomManyWithBias
where
    Iter: IntoIterator + ExactSizeIterator,
{
    type Output = Vec<<Iter as IntoIterator>::Item>;
    fn select_random_owned(self, rng: &mut ThreadRng, items: Iter) -> Self::Output {
        self.select_random_owned(rng, items)
    }
}

impl<T> SelectRandom<T> for SelectRandomManyWithBias {
    type Output<'a> = Vec<&'a T> where T: 'a;
    fn select_random<'a>(self, rng: &mut ThreadRng, items: &'a [T]) -> Self::Output<'a> {
        self.select_random(rng, items)
    }
}

impl<'a, T> SelectRandomMut<'a, &'a mut T> for SelectRandomManyWithBias
where
    T: 'a,
{
    type Output = Vec<&'a mut &'a mut T>;
    fn select_random_mut(self, rng: &mut ThreadRng, items: &'a mut [&'a mut T]) -> Self::Output {
        self.select_random_mut(rng, items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod select_random_with_bias {
        use rand::thread_rng;

        use super::*;

        #[test]
        fn test_select_random_owned() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let foo = Foo(1);
            let items = [foo];
            let selected = SelectRandomWithBias::new(Bias::Front)
                .select_random_owned(&mut thread_rng(), items.into_iter());
            assert_eq!(selected, Some(Foo(1)));
        }

        #[test]
        fn test_select_random() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let foo = Foo(1);
            let items = [&foo];
            let selected =
                SelectRandomWithBias::new(Bias::Front).select_random(&mut thread_rng(), &items);
            assert_eq!(selected, Some(&foo).as_ref());
        }

        #[test]
        fn test_select_random_mut() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let mut foo = Foo(1);
            let mut items = [&mut foo];
            let selected = SelectRandomWithBias::new(Bias::Front)
                .select_random_mut(&mut thread_rng(), &mut items);
            let Some(selected) = selected else {
                unreachable!();
            };
            selected.0 = 2;
            let selected_value = selected.0;
            let foo_value = foo.0;
            assert_eq!(selected_value, 2);
            assert_eq!(foo_value, 2);
        }
    }

    mod select_random_many_with_bias {
        use rand::thread_rng;

        use super::*;

        #[test]
        fn test_select_random_owned() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let items = [Foo(1), Foo(1), Foo(1)];
            let selected = SelectRandomManyWithBias::new(2, Bias::Front)
                .select_random_owned(&mut thread_rng(), items.into_iter());
            assert_eq!(selected, vec![Foo(1), Foo(1)]);
        }

        #[test]
        fn test_select_random() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let foo = Foo(1);
            let items = [&foo];
            let selected = SelectRandomManyWithBias::new(2, Bias::Front)
                .select_random(&mut thread_rng(), &items);
            assert_eq!(selected, vec![&&foo]);
        }

        #[test]
        fn test_select_random_mut() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let mut items = [&mut Foo(1), &mut Foo(1)];
            let mut selected = SelectRandomManyWithBias::new(2, Bias::Front)
                .select_random_mut(&mut thread_rng(), &mut items);
            {
                let Some(selected) = selected.get_mut(0) else {
                    unreachable!();
                };
                selected.0 = 2;
            }
            assert_eq!(selected.first(), Some(&&mut &mut Foo(2)));
        }
    }
}

use std::collections::HashSet;

use itertools::Itertools;

use crate::util::{Bias, random_index_bias};

pub trait SelectOther<T>: Copy {
    type Output;
    fn select_from<
        Iter: IntoIterator<Item = T, IntoIter = Iter2>,
        Iter2: Iterator<Item = T> + ExactSizeIterator,
    >(
        self,
        items: Iter,
    ) -> Self::Output;
}

pub trait SelectOtherRandom<T> {
    type Output;
    fn select_random<
        Iter: IntoIterator<Item = T, IntoIter = Iter2>,
        Iter2: Iterator<Item = T> + ExactSizeIterator,
    >(
        self,
        items: Iter,
    ) -> Self::Output;
}

#[derive(Debug, Copy, Clone, Default)]
pub struct SelectAll;

impl<T> SelectOther<T> for SelectAll {
    type Output = Vec<T>;

    fn select_from<
        Iter: IntoIterator<Item = T, IntoIter = Iter2>,
        Iter2: Iterator<Item = T> + ExactSizeIterator,
    >(
        self,
        items: Iter,
    ) -> Self::Output {
        items.into_iter().collect()
    }
}

#[derive(Debug, Copy, Clone, Default)]
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

impl<T> SelectOther<T> for SelectRandomWithBias {
    type Output = Option<T>;
    fn select_from<
        Iter: IntoIterator<Item = T, IntoIter = Iter2>,
        Iter2: Iterator<Item = T> + ExactSizeIterator,
    >(
        self,
        items: Iter,
    ) -> Self::Output {
        self.select_random(items)
    }
}

impl<T> SelectOtherRandom<T> for SelectRandomWithBias {
    type Output = Option<T>;
    fn select_random<
        Iter: IntoIterator<Item = T, IntoIter = Iter2>,
        Iter2: Iterator<Item = T> + ExactSizeIterator,
    >(
        self,
        items: Iter,
    ) -> Self::Output {
        let mut items = items.into_iter();
        items.nth(random_index_bias(items.len(), self.bias))
    }
}

#[derive(Debug, Copy, Clone, Default)]
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
    fn select_random_indexes(&self, len: usize) -> HashSet<usize> {
        let max_amount = self.amount.min(len);
        // not enough items just return the original slice as a new vec
        if max_amount >= len {
            (0..len).collect()
        } else if (max_amount as f32 / len as f32) < 0.5 {
            let mut selected_indexes = HashSet::new();
            while selected_indexes.len() < max_amount {
                selected_indexes.insert(random_index_bias(len, self.bias));
            }
            selected_indexes.into_iter().collect()
        } else {
            // it'll be faster to remove indexes randomly until we get the desired size
            let mut selected_indexes = (0..len).collect::<HashSet<_>>();
            while selected_indexes.len() > max_amount {
                selected_indexes.remove(&random_index_bias(len, self.bias.inverse()));
            }
            selected_indexes.into_iter().collect()
        }
    }
    fn select_random<
        T,
        Iter: IntoIterator<Item = T, IntoIter = Iter2>,
        Iter2: Iterator<Item = T> + ExactSizeIterator,
    >(
        self,
        items: Iter,
    ) -> Vec<T> {
        let mut items = items.into_iter();
        let (_, data) = self
            .select_random_indexes(items.len())
            .into_iter()
            .sorted()
            .fold((0, vec![]), |(skipped, mut data), next_ix| {
                let Some(item) = items.nth(next_ix - skipped) else {
                    unreachable!("index out of bounds?");
                };
                data.push(item);
                (next_ix + 1, data)
            });
        data
    }
}

impl<T> SelectOther<T> for SelectRandomManyWithBias {
    type Output = Vec<T>;
    fn select_from<
        Iter: IntoIterator<Item = T, IntoIter = Iter2>,
        Iter2: Iterator<Item = T> + ExactSizeIterator,
    >(
        self,
        items: Iter,
    ) -> Self::Output {
        self.select_random(items)
    }
}

impl<T> SelectOtherRandom<T> for SelectRandomManyWithBias {
    type Output = Vec<T>;
    fn select_random<
        Iter: IntoIterator<Item = T, IntoIter = Iter2>,
        Iter2: Iterator<Item = T> + ExactSizeIterator,
    >(
        self,
        items: Iter,
    ) -> Self::Output {
        self.select_random(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod select_random_with_bias {
        use super::*;

        #[test]
        fn test_select_random_owned() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let foo = Foo(1);
            let items = [foo];
            let selected = SelectRandomWithBias::new(Bias::Front).select_random(items);
            assert_eq!(selected, Some(Foo(1)));
        }

        #[test]
        fn test_select_random() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let foo = Foo(1);
            let items = &[foo];
            let selected = SelectRandomWithBias::new(Bias::Front).select_random(items);
            assert_eq!(selected, Some(Foo(1)).as_ref());
        }

        #[test]
        fn test_select_random_mut() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let mut foo = Foo(1);
            let items = [&mut foo];
            let selected = SelectRandomWithBias::new(Bias::Front).select_random(items);
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
        use super::*;

        #[test]
        fn test_select_random_indexes() {
            let items = 0..1000000;
            let select = SelectRandomManyWithBias::new(items.len() - 1, Bias::Front)
                .select_random_indexes(items.len());
            assert_eq!(select.len(), items.len() - 1);
        }

        #[test]
        fn test_select_random_owned() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let items = [Foo(1), Foo(1), Foo(1)];
            let selected = SelectRandomManyWithBias::new(2, Bias::Front).select_random(items);
            assert_eq!(selected, vec![Foo(1), Foo(1)]);
        }

        #[test]
        fn test_select_random() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let foo = Foo(1);
            let items = [&foo];
            let selected = SelectRandomManyWithBias::new(2, Bias::Front).select_random(items);
            assert_eq!(selected, vec![&foo]);
        }

        #[test]
        fn test_select_random_range() {
            let selected = SelectRandomManyWithBias::new(8, Bias::Front).select_random(0..10);
            assert_eq!(selected.into_iter().collect::<HashSet<_>>().len(), 8);
        }

        #[test]
        fn test_select_random_range_a() {
            let len = 50000;
            let expected = len / 2 - 1;
            let selected =
                SelectRandomManyWithBias::new(expected, Bias::Front).select_random(0..len);
            assert_eq!(selected.into_iter().collect::<HashSet<_>>().len(), expected);
        }

        #[test]
        fn test_select_random_range_b() {
            let len = 50000;
            let expected = len / 2 + 1;
            let selected =
                SelectRandomManyWithBias::new(expected, Bias::Front).select_random(0..len);
            assert_eq!(selected.into_iter().collect::<HashSet<_>>().len(), expected);
        }

        #[test]
        fn test_select_random_mut() {
            #[derive(Debug, PartialEq)]
            struct Foo(usize);
            let items = [&mut Foo(1), &mut Foo(1)];
            let mut selected = SelectRandomManyWithBias::new(2, Bias::Front).select_random(items);
            {
                let Some(selected) = selected.get_mut(0) else {
                    unreachable!();
                };
                selected.0 = 2;
            }
            assert_eq!(selected.into_iter().next(), Some(&mut Foo(2)));
        }
    }
}

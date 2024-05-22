use crate::util::Odds;

#[derive(Debug, Copy, Clone)]
pub enum Probability {
    /// Represents a probability of 0.0 (never happening).
    Never,
    /// Represents a probability with given odds (0.0 to 1.0).
    Some(f64),
    /// Represents a probability of 1.0 (guaranteed to happen).
    Guaranteed,
}

impl Probability {
    /// Converts the `Probability` to a `f64` value.
    /// - `Never` returns 0.0.
    /// - `Some(odds)` returns the value of `odds`.
    /// - `Guaranteed` returns 1.0.
    pub fn as_f64(&self) -> f64 {
        match self {
            Probability::Never => 0.0,
            Probability::Some(odds) => {
                debug_assert!(
                    self.is_valid(),
                    "{odds} outside of expected range 0.0..=1.0"
                );
                *odds
            }
            Probability::Guaranteed => 1.0,
        }
    }
    pub fn assert_is_valid(&self) {
        assert!(
            self.is_valid(),
            "{} outside of expected range 0.0..=1.0",
            self.as_f64()
        );
    }
    pub fn is_valid(&self) -> bool {
        match self {
            Probability::Some(odds) => Self::is_valid_odds(*odds),
            Probability::Never | Probability::Guaranteed => true,
        }
    }
    pub fn is_valid_odds(odds: Odds) -> bool {
        (0.0..=1.0).contains(&odds)
    }
    pub fn assert_is_valid_odds(odds: Odds) -> Odds {
        assert!(
            Self::is_valid_odds(odds),
            "{odds} outside of expected range 0.0..=1.0"
        );
        odds
    }
}

impl PartialEq for Probability {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Never, Self::Never) | (Self::Guaranteed, Self::Guaranteed) => true,
            _ => self.as_f64() == other.as_f64(),
        }
    }
}

impl From<Probability> for f64 {
    fn from(value: Probability) -> Self {
        value.as_f64()
    }
}

impl From<f64> for Probability {
    fn from(value: f64) -> Self {
        match value {
            0.0 => Self::Never,
            1.0 => Self::Guaranteed,
            odds => Self::Some(Probability::assert_is_valid_odds(odds)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_f64() {
        assert_eq!(Probability::Never.as_f64(), 0.0);
        assert_eq!(Probability::Some(0.5).as_f64(), 0.5);
        assert_eq!(Probability::Guaranteed.as_f64(), 1.0);
    }

    #[test]
    fn test_partial_eq() {
        assert_eq!(Probability::Never, Probability::Never);
        assert_eq!(Probability::Some(0.5), Probability::Some(0.5));
        assert_eq!(Probability::Guaranteed, Probability::Guaranteed);
        assert_eq!(Probability::Guaranteed, Probability::Some(1.0));
        assert_eq!(Probability::Never, Probability::Some(0.0));
        assert_ne!(Probability::Some(0.5), Probability::Guaranteed);
    }

    #[test]
    fn test_from() {
        let value: f64 = Probability::Some(0.7).into();
        assert_eq!(value, 0.7);

        let prob: Probability = 0.7.into();
        assert_eq!(prob, Probability::Some(0.7));

        let prob: Probability = 1.0.into();
        assert_eq!(prob, Probability::Guaranteed);

        let prob: Probability = 0.0.into();
        assert_eq!(prob, Probability::Never);
    }

    #[test]
    fn test_is_valid() {
        assert!(Probability::Never.is_valid());
        assert!(Probability::Some(0.5).is_valid());
        assert!(Probability::Guaranteed.is_valid());
        assert!(!Probability::Some(1.5).is_valid());
        assert!(!Probability::Some(-0.1).is_valid());
    }

    #[test]
    #[should_panic(expected = "outside of expected range 0.0..=1.0")]
    fn test_assert_is_valid() {
        Probability::Some(1.5).assert_is_valid();
    }

    #[test]
    fn test_is_valid_odds() {
        assert!(Probability::is_valid_odds(0.0));
        assert!(Probability::is_valid_odds(0.5));
        assert!(Probability::is_valid_odds(1.0));
        assert!(!Probability::is_valid_odds(1.5));
        assert!(!Probability::is_valid_odds(-0.1));
    }

    #[test]
    #[should_panic(expected = "outside of expected range 0.0..=1.0")]
    fn test_assert_is_valid_odds() {
        Probability::assert_is_valid_odds(1.5);
    }
}

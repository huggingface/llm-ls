fn sum(lhs: i32, rhs: i32) -> i32 {
    lhs + rhs
}

fn sub(lhs: i32, rhs: i32) -> i32 {
    lhs - rhs
}

fn mul(lhs: i32, rhs: i32) -> i32 {
    lhs * rhs
}

fn div(lhs: i32, rhs: i32) -> i32 {
    lhs / rhs
}

fn main() {
    println!("42 + 42 = {}", sum(42, 42));
    println!("41 - 42 = {}", sub(41, 42));
    println!("42 * 42 = {}", mul(42, 42));
    println!("42 / 42 = {}", div(42, 42));
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_sum() {
        assert_eq!(42 + 42, super::sum(42, 42));
    }

    #[test]
    fn test_sub() {
        assert_eq!(42 - 42, super::sub(42, 42));
        assert_eq!(41 - 42, super::sub(41, 42));
    }

    #[test]
    fn test_mul() {
        assert_eq!(42 * 42, super::mul(42, 42));
    }

    #[test]
    fn test_div() {
        assert_eq!(42 / 42, super::div(42, 42));
    }
}

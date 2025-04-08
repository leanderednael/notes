# [Functional Programming](https://www.edx.org/learn/computer-programming/delft-university-of-technology-introduction-to-functional-programming)

Functional programming is a style of programming in which the basic method of computation is the application of functions to arguments.

Variable assignment vs. function application:

```java
int total = 0;
for (int i = 0; i < 10; i++) { total += i; }
```

```haskell
sum [1..10]
```

Lisp was the first functional programming language, with some influences from lambda calculus, but retaining variable assignment.

## Haskell

GHC (Glasgow Haskell Compiler) is the leading compiler for Haskell, and comes with an interactive shell called GHCi.

| Mathematics       | Haskell         |
| ----------------- | --------------- |
| $f(a, b) + c * d$ | `f a b + c * d` |
| $f(x)$            | `f x`           |
| $f(x, y)$         | `f x y`         |
| $f(g(x))$         | `f (g x)`       |
| $f(x, g(y))$      | `f x (g y)`     |
| $f(x) g(y)$       | `f x * g y`     |

Common functions on strings:

`length "Hello"` returns `5`.

`head "Hello"` returns `'H'`.

`tail "Hello"` returns `"ello"`.

`last "Hello"` returns `'o'`.

`init "Hello"` returns `"Hell"`.

`reverse "Hello"` returns `"olleH"`.

`take 3 "Hello"` returns `"Hel"`.

`drop 3 "Hello"` returns `"lo"`.

`take 3 "Hello"` returns `"Hel"`.

`drop 3 "Hello"` returns `"lo"`.

## Types and Classes

A type is a name for a collection of related values.

All type errors are found at compile time, which makes programs safer and faster by removing the need for type checks at runtime.

Haskell basic types:

- `Bool` (True, False)
- `Char` (a single character)
- `String` (a sequence of characters)
- `Int` (a fixed-precision whole number)
- `Integer` (an arbitrary-precision whole number)
- `Float` (a decimal number)
- `Double` (a double-precision decimal number)

Furthermore, there are list, tuple and function types.

**Curried** functions (invented by Haskell Curry) take one argument at a time.

```haskell
add  :: (Int,Int) -> Int

add' :: Int -> (Int -> Int)
```

Curried functions are more flexible than functions on tuples, because useful functions can often be made by partially applying a curried function. Unless tupling is explicitly required, all functions in Haskell are normally defined in curried form.

A function is called **polymorphic** ("of many forms") if it is defined in a generic way so that it can accept arguments of different types.

A polymorphic function is called overloaded if its type contains one or more class constraints.

**Type classes** are used to create a common interface for different types.

- `Num` for numeric types, e.g. `sum :: Num a => [a] -> a`
- `Eq` for types that support equality testing, e.g. `== :: Eq a => a -> a -> Bool`
- `Ord` for types that support ordering, e.g. `< :: Ord a => a -> a -> Bool`

## Functions, Operators and Sections

**Conditional expressions, guards and pattern matching** are alternative ways to express logical conditions in functions.

**Lambdas** are useful because they can give a formal meaning to functions defined using currying, functions that return functions as results, and functions that are only referenced once.

Operators in infix notation can be transformed into prefix notation (so-called **sections**) by surrounding them with parentheses. Sections can also include an argument (they are partially applied) by including one argument within the parentheses.

```haskell
1+2

(+) 1 2

(1+) 2

(+2) 1
```

Sections are useful for creating new functions from existing operators.

- Successor function: `(1+)` is a function that takes an argument, adds 1 to it, and returns the result.
- Predecessor function: `(-1)` is a function that takes an argument, subtracts 1 from it, and returns the result.
- Reciprocal function: `(1/)` is a function that takes an argument, divides 1 by it, and returns the result.
- Double function: `(*2)` is a function that takes an argument, multiplies it by 2, and returns the result.
- Halve function: `(/2)` is a function that takes an argument, divides it by 2, and returns the result.

## List Comprehensions

| Mathematics (set comprehension)  | Haskell (list comprehension) |
| -------------------------------- | ---------------------------- |
| $\{ x^2 \mid x \in \{1...5\} \}$ | `[x^2 \| x <- [1..5]]`       |

The expression `[1..5]` is called a **generator** as it states how to generate values for `x`.

List comprehensions can use **guards** to restrict the values produced by earlier generators.

```haskell
[x | x <- [1..10], even x]
```

**Zip** is a useful library function that maps two lists into a single list of pairs of their corresponding elements.

Because **strings** are just a special kind of lists (i.e. of characters), any polymorphic function that operates on lists can also be applied to strings.

## Recursive Functions

In Haskell all repetition is expressed using recursion.

Properties of functions defined using recursion can be proven using mathematical **induction**.

The **quicksort** algorithm for sorting a list of integers can be specified by two rules: (i) the empty list is already sorted; (ii) non-empty lists can be sorted by sorting the tail values less than or equal to the head, sorting the tail values greater than the head, and then appending the resulting lists on either side of the head value.

## Higher-Order Functions

A **higher-order function** is a function that takes one or more functions as arguments or returns a function as its result.

Examples:

- `map`
- `filter`
- `foldr`
- `.` returns the **composition** of two functions as a single function. It is wise to use function composition sparingly as it makes the code less readable.
- `all` decides if every element of a list satisfies a given predicate.
- `takeWhile` selects all elements from a list while a predicate holds whereas the `dropWhile` function removes all elements while a predicate holds.

Higher-order functions are the ultimate solution to DRY.

## Functional Parsers and Monads

A **parser** analyses a piece of text to determine its syntactic structure. For example, GHC parses Haskell source code, Unix parses shell commands, and web browsers parse HTML.

`2*3+4` can be parsed to the following tree:

```text
      +
     / \
    *   4
   / \
  2   3
```

In a functional language such as Haskell, parsers can naturally be viewed as functions that transform a string into a structured representation of the string's syntactic structure.

A string might be parsable in many ways, incl. none, so we generalise to a list of results, each of which is a parse tree.

```haskell
type Parser = String -> [(Tree, String)]
```

A parser might not always produce a tree, so we generalise to a value of any type.

```haskell
type Parser a = String -> [(a, String)]
```

The Parser type is a **monad**, a mathematical structure that has proven useful for modelling many different kinds of computations.

A sequence of parsers can be combined as a single composite parser using the keyword `do.` Following the layout rule, each parser must begin in the same column. The `do` notation is not specific to the Parser type, but can be used with any monadic type. The main advantage of monads is that you can use the `do` notation.

## Interactive Programs (I/O)

Haskell programs are pure mathematical functions. However, reading from the keyboard and writing to the screen are side effects.

Interactive programs can be written by using types to distinguish pure expressions from impure actions that may involve side effects.

For example, an expression of type `IO String` denotes a possibly side-effecting computation that, if it terminates, produces a value of type `String`.

The standard library provides a number of **imperative** actions, including the following primitives:

- `getChar :: IO Char` reads a single character from the keyboard.
- `putChar :: Char -> IO ()` writes a single character to the screen, and returns nothing.
- `getLine :: IO String` reads a line of text from the keyboard.
- `putStr :: String -> IO ()` writes a string to the screen, and returns nothing.

A **sequence** of actions can be combined using the `do` notation.

```haskell
main :: IO ()
main = do
  putStr "What is your name? "
  name <- getLine
  putStrLn ("Hello, " ++ name ++ "!")
```

## Declaring Types and Classes

In Haskell, a new name for an existing type can be defined using a **type declaration**.

```haskell
type String = [Char]
```

Like function definitions, type declarations can also have **parameters**.

```haskell
type Pair a = (a, a)

type Position = Pair Int
```

A completely new type can defined by specifying its values using a **data declaration**.

```haskell
data Bool = False | True
```

The two values of type `Bool` are called the **constructors** of the type.

Type and constructor names must start with a capital letter.

The constructors of a data declaration can also have parameters. For example, `data Shape = Circle Float | Rect Float Float`.

In Haskell, types can be **recursive**.

## Lazy Evaluation

What distinguishes Haskell from other languages with higher-order functions is that Haskell uses lazy evaluation to evalute expressions.

Using lazy evaluation, expressions are only evaluated when needed to produce the result of a computation, which often is rather handy if you want to write performant code. It

- avoids unnecessary computations
- allows programs to be more modular
- allows us to program with infinite data structures

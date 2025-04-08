# Software Development

## Software Development Resources

### Dependency Management

- [Poetry](https://muttdata.ai/blog/2020/08/21/a-poetic-apology.html)
- [Poetry with Private Repos (e.g. Artifactory)](https://github.com/python-poetry/poetry/issues/4389)

```bash
private-package = {version = "^0.1.0", source = "artifactory"}

[[tool.poetry.source]]
name = "artifactory"
secondary = true
url = "https://artifactory.whatever.com/artifactory/some-repository/"
```

### Data Versioning

- [Build a Reproducible and Maintainable Data Science Project](https://khuyentran1401.github.io/reproducible-data-science)
- [Data Version Control](https://mathdatasimplified.com/2023/02/20/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-2/)
- [Git-LFS](https://www.youtube.com/watch?v=xPFLAAhuGy0)

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.python.sh | bash
pip install git-lfs

git init
git lfs install

git lfs track "*.pickle"
```

### Numerical Computing Techniques

- [Iterating over pd.DataFrame](https://towardsdatascience.com/efficiently-iterating-over-rows-in-a-pandas-dataframe-7dd5f9992c01)

### RegEx

- [English to RegEx](https://www.autoregex.xyz/)

### Testing

- [Pytest](https://towardsdatascience.com/pytest-for-data-scientists-2990319e55e6)
- [mock and patch](https://write.agrevolution.in/python-unit-testing-mock-and-patch-8ba9c796c9c2)
- Test Doubles:
  - A **dummy** is a placeholder to fill some required parameters.
  - A **fake** simulates a real dependency with a simplified working implementation.
  - A **stub** behaves exactly as instructed.
  - A **spy** records interactions with the production code, allowing tests to extract the interaction history when required.
  - A **mock** additionally includes assertions (expectations) about the interactions with the production code.

### Software Design

- [Clean Code](https://testdriven.io/blog/clean-code-python/)
- [OOP](https://www.pythontutorial.net/python-oop/)
- [Inheritance and Composition](https://realpython.com/inheritance-composition-python/)
- [Domain-Driven Design](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/ddd-oriented-microservice):
  - Ubiquitous Language: Creating a shared vocabulary and terminology that both developers and domain experts can use to discuss the domain.
  - Bounded Contexts: Defining specific boundaries around parts of the domain to isolate them and manage complexity.
  - Entities and Value Objects: Modeling domain entities and value objects that represent key concepts in the domain.
  - Aggregates: Clustering related entities and value objects together to enforce consistency and invariants.
  - Repositories: Providing a structured way to access and persist domain objects.
- [Hexagonal Architecture (a.k.a. Ports & Adapters)](https://alistair.cockburn.us/hexagonal-architecture/):
  - separate business logic (the hexagon) from its external dependencies (the ports & adapters)
  - this separation allows for improved flexibility and testability
- Event Sourcing:
  - Events: Immutable records of something that has happened in the system. Events capture both state changes and the intent behind those changes.
  - Event Store: A data store that stores the events in an append-only fashion.
  - Projections: Mechanisms to build read models (current state) from the event stream for querying purposes.
  - Event Handlers: Components responsible for updating the read models in response to events.
- Command-Query Responsibility Segretation (CQRS):
  - Command Handlers: Components responsible for processing commands and updating the write-side (event sourcing) of the system.
  - Query Handlers: Components responsible for handling queries and providing data from the read-side (projections) of the system.
  - Command and Query Models: These are often distinct from each other, tailored to the specific needs of commands and queries.

#### APIs

- [FastAPI ML Serving](https://luis-sena.medium.com/how-to-optimize-fastapi-for-ml-model-serving-6f75fb9e040d)
- [FastAPI Design Patterns](https://theprimadonna.medium.com/5-must-know-design-patterns-for-building-scalable-fastapi-applications-36f9f31059fd)
- [FastAPI Three-Layer Architecture](https://medium.com/@yashika51/write-robust-apis-in-python-with-three-layer-architecture-fastapi-and-pydantic-models-3ef20940869c)
- [Metaprogramming with Decorators](https://medium.com/@angusyuen/writing-maintainable-pythonic-code-metaprogramming-with-decorators-2fc2f1d358db)

## Clean Code

### Chapter 1: Clean Code

Beautiful code makes the language look like it was made for the problem: It is not the language that makes programmes appear simple. It is the programmer that makes the language appear simple!

#### We are Authors

The `@author` field of a Javadoc tells us who we are.

#### The Boy Scout Rule

### Chapter 2: Meaningful Names

#### Use Intention-Revealing Names

#### Avoid Disinformation

#### Make Meaningful Distinctions

#### Use Pronounceable Names

#### Use Searchable Names

### Chapter 3: Functions

#### Small

The indent level of a function should not be greater than one or two. The maximum length of a function should be around 20 lines.

#### Don One Thing

Functions should do one thing. They should do it well. They should do it only.

Functions that do one thing cannot be reasonably divided into sections.

#### One Level of Abstraction per Function

_The Stepdown Rule:_ We want every function to be followed by those at the next level of abstraction so that we can read the programme, descending one level of abstraction at a time as we read down the list of functions.

#### Switch Statements

`switch` statements by definition do more than one thing. Unfortunately, we can't always avoid `switch` statements, but we can make sure that each `switch` statement is buried in a low-level class and is never repeated.

`switch` statements are tolerable if they appear only once, are used to create polymorphic objects, and are hidden behind an inheritance relationship so the rest of the system can't see them; e.g. an abstract factory.

#### Use Descriptive Names

#### Function Arguments

The ideal number of arguments for a function is zero (niladic). Next comes one (monadic), followed closely by two (dyadic). Three arguments (triadic) should be avoided where possible. More than three (polyadic) requires very special justification—and then shouldn't be used anyway.

##### Flag Arguments

Flag arguments are ugly. Passing a boolean into a function is a truly terrible practice. A function should do one thing. Therefore, a function with a boolean argument does two things. It should be split in two.

##### Argument Objects

Create objects from cohesive groups of arguments. This reduces the number of arguments and makes the code more readable.

#### Have No Side Effects

Side effects create strange temporal couplings.

#### Command Query Separation

Functions should either do something or answer something, but not both. Either your function should change the state of an object, or it should return some information about that object.

#### Prefer Exceptions to Returning Error Codes

If you use exceptions instead of returned error codes, then the error processing code can be separated from the happy path code and can be simplified.

Error Handling is one thing. Business Logic is another. Don't mix them.

Open-Closed Principle: When you use exceptions, then new exceptions are derivatives of the base exception class. This means that you can add new exceptions without changing existing code.

Functions are the verbs of the language, and classes are the nouns.

### Chapter 4: Comments

#### Explain Yourself in Code

Comments do not make up for bad code. If the code is hard to understand, then rewrite it.

#### Good Comments

- Legal Comments
- Warning of Consequences
- TODO Comments
- DocStrings

### Chapter 5: Formatting

### Chapter 6: Objects and Data Structures

#### Data Abstraction

#### Data/Object Anti-Symmetry

Objects hide their data behind abstractions and expose functions that operate on that data.

Data structures expose their data and have no meaningful functions.

They are virtual opposites.

Procedural code (code using data structures) makes it easy to add new functions without changing the existing data structures. OO code, on the other hand, makes it easy to add new classes without changing existing functions.

The complement is also true: Procedural code makes it hard to add new data structures because all the functions must change. OO code makes it hard to add new functions because all the classes must change.

In any complex system there are going to be times when we want to add new data types rather than new functions. For these cases, objects and OO are most appropriate. On the other hand, there will also be times when we'll want to add new functions as opposed to data types. In that case, procedural code and data structures will be more appropriate.

#### The Law of Demeter

#### Data Transfer Objects

The quintessential form of a data structure is a class with public variables and no functions. This is sometimes called a DTO. DTOs are very useful structures, especially when communicating with databases or parsing messages from sockets, and so on. They often become the first in a series of translation stages that convert raw data in a database into objects in the application.

_Active records_ are special forms of DTOs. They are data structures with public (or bean-accessed) variables; but they typically have navigational methods like `save` and `find`. Typically, these Active Records are direct translations from database tables, or other data sources. They are however not objects, they are data structures. Separate objects must be created to represent the business logic of the application.

### Chapter 7: Error Handling

#### Use Exceptions Rather Than Return Codes

#### Write Your Try-Catch-Finally Statement First

#### Use Unchecked Exceptions

The price of checked exceptions is an Open/Closed Principle violation. If you throw a checked exception from a method in your code and the `catch` is three levels above, you must declare that exception in the signature of each method between you and the `catch`.

#### Provide Context with Exceptions

#### Define Exception Classes in Terms of a Caller's Needs

#### Define the Normal Flow

#### Don't Return Null

#### Don't Pass Null

### Chapter 8: Boundaries

We manage third-party boundaries by having very few places in the code that refer to them, ideally wrapped in an Adapter class.

### Chapter 9: Unit Tests

#### The Three Laws of TDD

##### The First Law: You May Not Write Production Code Until You Have Written a Failing Unit Test

##### The Second Law: You May Not Write More of a Unit Test Than Is Sufficient to Fail, and Not Compiling Is Failing

##### The Third Law: You May Not Write More Production Code Than Is Sufficient to Pass the Currently Failing Test

#### Keep Tests Clean

The problem is that tests must change as the production code evolves. The dirtier the tests, the harder they are to change.

#### Clean Tests

#### One Assert per Test

#### F.I.R.S.T

_Fast:_ Tests should be fast. If they are slow, then you'll be less likely to run them.

_Independent:_ Tests should not depend on each other running in a particular order to pass, or on any state that is shared between them.

_Repeatable:_ Tests should be repeatable in any environment. This means that they should not depend on any external system like a database, a file system, or a network connection.

_Self-Validating:_ Tests should have boolean output: either pass or fail. They should not output anything else like logging or print to a screen.

_Timely:_ Tests should be written in a timely fashion. The sooner you write the test, the sooner you'll find the defect.

### Chapter 10: Classes

Functions are the verbs, classes the nouns of our language.

#### Class Organization

##### Encapcsulation

There is seldom a good reason to have a public variable.

We like to put the private utilities called by a public function right after the public function itself. This follows the stepdown rule and helps the programme read like a newspaper article.

#### Classes Should be Small

##### The Single Responsibility Principle

A class or module should have one, and only one, reason to change.

Each small class encapsulates a single responsibility, has a single reason to change, and collaborates with a few other classes to achieve the desired functionality of the system.

##### Cohesion

Classes should have a small number of instance variables. Each of the methods of a class should manipulate one or more of those variables.

A class in which each variable is used by each method is maximally cohesive.

##### Maintaining Cohesion Results in Many Small Classes

#### Organizing for Change

_Open/Closed Principle:_ Classes should be open for extension, but closed for modification.

_Dependency Inversion Principle:_ A client class depending upon concrete details is at risk when those details change. Instead of depending upon concrete details, the client should depend upon an abstraction / interface.

### Chapter 11: Systems

#### Separate Constructing a System from Using It

Software systems should separate the startup process, when the application objects are constructed and the dependencies are wired together, from the runtime logic that takes over after startup.

_Lazy Initialization / Evaluation:_ Doesn't incur the overhead of object construction until it is needed.

##### Separation of Main

One way to separate construction from use is simply to move all aspects of construction to `main`, or modules called by `main`, and to design the rest of the system assuming that all objects have been constructed and wired up appropriately.

The flow of control is easy to follow. The `main` function builds the objects necessary for the system, then passes them to the application, which simply uses them. Notice the direction of the dependency arrows crossing the barrier between `main` and the application. They all go one direction, pointing away from `main`. This means that the application has no knowledge of `main` or of the construction process. It simply expects that everything has been built properly.

##### Factories

Sometimes, of course, we need to make the applicaiton responsible for _when_ an object gets created. In this case, we can use the Abstract Factory pattern to give the application control of when to build the objects, but keep the details of that construction separate from the application code.

##### Dependency Injection

A powerful mechanism for separating construction from use is dependency injection, the application of the _Inversion of Control_ principle to dependency management. Inversion of Control moves secondary responsibilities from an object to other objects that are dedicated to the purpose, thereby supporting the Single Responsibility Principle. Because setup is a global concern, this authoritative mechanism will usually be either the `main` routine or a special-purpose _container._

During the construction process, the DI container instantiates the required objects (usually on demand) and uses the constructor arguments or setter methods provided to wire together the dependencies. Which dependent objects are actually used is specified through a configuration file, or programmatically in a special-purpose construction module.

#### Scaling Up

Software systems are unique compared to physical systems. Their architectures can grow incrementally, _if_ we maintain the proper separation of concerns.

#### Systems Need Domain-Specific Languages

If you are implementing domain logic in the same language that a domain expert uses, there is less risk that you will incorrectly translate the domain into the implementation. DSLs, when used effectively, raise the abstraction level above code idioms and design patterns.

### Chapter 12: Emergence

#### Simple Design Rule 1: Runs All the Tests

#### Simple Design Rule 2-4: Refactoring

The fact that we have these tests eliminates the feat that cleaning up the code will break it!

We can increase cohesion, decrease coupling, separate concerns, modularise system concerns, shrink our functions and classes, choose better names, and so on.

##### No Duplication

The Template Method pattern is a common technique for removing higher-level duplication.

##### Expressive

By using the standard pattern names, such as command or visitor, in the names of the classes that implement those patterns, you can succinctly describe your design to other developers.

Well-written tests are also expressive. A primary goal of tests is to act as documentation by example.

##### Minimal Classes and Methods

### Chapter 13: Concurrency

Objects are abstractions of processing. Threads are abstractions of schedule. - James O. Coplien

Concurrency is a decoupling strategy, and allows for higher throughput and lower response times.

**Misconceptions:**

- Concurrency always improves performance. It does not.
- Design does not change when writing concurrent code. It does.

#### Concurrency Defence Principles

##### Single Responsibility Principle

Keep concurrency-related code separate from other code.

##### Corollary: Limit the Scope of Data

Take data encapsulation to heart; severely limit the access of any data that may be shared.

One solution is to use the `synchronized` keyword to protect a critical section in the code that uses the shared object.

It is important to restrict the number of such critical sections. The fewer places shared data can get updated, the better.

##### Corollary: Use Copies of Data

A good way to avoid shared data issues is to avoid sharing the data in the first place. In some situations it is possible to copy objects and treat them as read-only. In other cases it might be possible to copy objects, collect results from multiple threads in these copies and then merge the results in a single thread.

If using copies of objects allows the code to avoid synchronizing, the savings in avoiding the intrinsic lock will likely make up for the additional creation and garbage collection overhead.

##### Corollary: Threads Should Be as Independent as Possible

Attempt to partition data into independent subsets that can be operated on by independent threads, possibly in different processors.

#### Know Your Library

- thread-safe collections
- executor framework for executing unrelated tasks
- non-blocking solutions when possible
- library modules that are not thread-safe

#### Know Your Execution Models

- bound resources
- mutual exclusion
- starvation
- deadlock
- livelock

Most concurrency problems tend to be some combination of these three problems:

- Producer-Consumer
- Readers-Writers
- Dining Philosophers
-

#### Keep Synchronised Sections Small

### Chapter 14: Successive Refinement

#### First Make It Work

#### Then Make It Right

### Chapter 17: Smells and Heuristics

#### Comments

##### C1: Inappropriate Information

##### C2: Obsolete Comment

##### C3: Redundant Comment

##### C4: Poorly Written Comments

##### C5: Commented-Out Code

#### Environment

##### E1: Build Requires More Than One Step

##### E2: Tests Require More Than One Step

#### Functions

##### F1: Too Many Arguments

##### F2: Output Arguments

##### F3: Flag Arguments

##### F4: Dead Function

#### General

##### G1: Multiple Languages in One Source File

##### G2: Obvious Behaviour Is Unimplemented

##### G3: Incorrect Behaviour at the Boundaries

##### G4: Overridden Safeties

##### G5: Duplication

##### G6: Code at Wrong Level of Abstraction

##### G7: Base Classes Depending on Their Derivatives

##### G8: Too Much Information

##### G9: Dead Code

##### G10: Vertical Separation

##### G11: Inconsistency

##### G12: Clutter

##### G13: Artificial Coupling

##### G14: Feature Envy

##### G15: Selector Arguments

##### G16: Obscured Intent

##### G17: Misplaced Responsibility

##### G18: Inappropriate Static

##### G19: Use Explanatory Variables

##### G20: Functions Should Say What They Do

##### G21: Understand the Algorithm

##### G22: Make Logical Dependencies Physical

##### G23: Prefer Polymorphism to If/Else or Switch/Case

##### G24: Follow Standard Conventions

##### G25: Replace Magic Numbers with Named Constants

##### G26: Be Precise

##### G27: Structure over Convention

##### G28: Encapsulate Conditionals

##### G29: Avoid Negative Conditionals

##### G30: Functions Should Do One Thing

##### G31: Hidden Temporal Couplings

##### G32: Don't Be Arbitrary

##### G33: Encapsulate Boundary Conditions

##### G34: Functions Should Descend Only One Level of Abstraction

##### G35: Keep Configurable Data at High Levels

##### G36: Avoid Transitive Navigation

#### Java

##### J1: Avoid Long Import Lists by Using Wildcards

##### J2: Don't Inherit Constants

##### J3: Constants versus Enums

#### Names

##### N1: Choose Descriptive Names

##### N2: Choose Names at the Appropriate Level of Abstraction

##### N3: Use Standard Nomenclature Where Possible

##### N4: Unambiguous Names

##### N5: Use Long Names for Long Scopes

##### N6: Avoid Encodings

##### N7: Names Should Describe Side Effects

#### Tests

##### T1: Insufficient Tests

##### T2: Use a Coverage Tool

##### T3: Don't Skip Trivial Tests

##### T4: An Ignored Test Is a Question about an Ambiguity

##### T5: Test Boundary Conditions

##### T6: Exhaustively Test Near Bugs

##### T7: Patterns of Failure are Revealing

##### T8: Test Coverage Patterns Can Be Revealing

##### T9: Tests Should Be Fast

## Clean Architecture

### Part I: Introduction

#### What is Design and Architecture?

The low-level details and the high-level structure are all part of the same whole. They form a continuous fabric that defines the shape of the system. There is simply a continuum of decisions from the highest to the lowest levels.

The goal of software architecture is to minimise the human resources required to build and maintain the required system.

#### A Tale of Two Values

Function or architecture? Is it more important for the software system to work, or is it more important for the software system to be easy to change?

It is the responsibility of the software development team to assert the importance of architecture over the urgency of features.

### Part II: Programming Paradigms

A paradim tells you which programming structures to use, and when to use them. To date, there have been three such paradigms. For reasons we shall discuss later, there are unlikely to be any others.

#### Paradigm Overview

_Structured programming imposes discipline on direct transfer of control._ Dijkstra replaced unrestrained goto statements with structured control constructs such as if, while, and for. These constructs are the foundation of all modern programming languages.

_Object-oriented programming imposes discipline on indirect transfer of control._ Two programmers noticed that the function call stack frame could moved to a heap, thereby allowing local variables declared by a function to exist long after the function returned. The function became a constructor for a class, the local variables became instance variables, and the nested functions became methods.

_Functional programming imposes discipline upon assignment._ A foundational notion of $\lambda$-calculus is immutability - that is, the notion that the values of symbols do not change. This effectively means that a functional language has no assignment statement. Most functional languages do, in fact, have some means to alter the value of a variable, but only under very strict discipline.

Each of these paradigms removes capabilities from the programmer. None of them adds new capabilities. The three paradigms together remove `goto` statements, function pointers, and assignment. Is there anything left to take away? Probably not.

In software architecture, we use polymorphism as the mechanism to cross architectural boundaries; we use functional programming to impose discipline on the location of and access to data; and we use structured programming as the algorithmic foundation of our modules. Notice how well these three align with _z_

#### Structured Programming

Dijkstra realized that modules that used only `if/then/else` and `do/while` control structures could be recursively subdivided into smaller provable units. Two years later it was proved that all programmes can be constructed from just three structures: _sequence, selection, and iteration._ These three structures are the foundation of all modern programming languages.

Testing shows the presence, not the absence, of bugs. Software is like a science. We show correctness by failing to prove incorrectness, despite our best effort. Software architects strive to define modules, components and services that are easily falsifiable (testable).

#### Object-Oriented Programming

##### Encapsulation

In C, header files provided perfect encapsulation. But then came OO C++, where for technical reasons, member variables of a class need to be declared in the header file. Thus, the languages that claim to provide OO have in fact only weakened the once perfect encapsulation we enjoyed with C. The way encapsulation is partially repaired is by introducing the `public`, `private` and `protected` keywords into the language.

##### Inheritance

Inheritance is simply the redeclaration of a group of variables and functions within an enclosing scope. This is something that C programmers were able to do before already as well, albeit manually. The only difference is that now the compiler does it for you.

##### Polymorphism

Polymorphism is an application of pointers to functions, which again, also already existed in C. Pointers to functions are dangerous, and OO languages eliminate these conventions.

Moreover, in C the flow of control was dictated by the behaviour of the system, and the source code dependencies were dictated by that flow of control. Now, with OO, through dependency inversion, there is inversion of control - the flow of control is dictated by the source code dependencies.

Thus, _OO imposes discipline on indirect transfer of control: It allows the plugin architecture to be used anywhere, for anything._

Through the use of polymorphism, there is absolute control over every source code dependency in the system. It allows the architect to create a plugin architecture, in which modules that contain high-level policies are independent of modules that contain low-level details.

#### Functional Programming

##### Immutability

In functional programming, variables are initialized but never changed.

_You cannot have a race condition or a concurrent update problem if no variable is ever updated. You cannot have deadlocks without mutable locks. All the problems that we face in concurrent applications cannot happen if there are no mutable variables._

##### Segregation of Mutability

Without infinite storage and infinite processor speed, immutability is not that practicable. One of the most common compromises is to segregate the application into mutable and immutable components.

##### Event Sourcing

Now imagine that instead of storing the account balances, we store only the transactions. Whenever someone wants to know the balance of an account, we simply add up all the transactions for that account.

Event sourcing is a strategy wherein we store the transactions, but not the state.When state is required, we simply apply all the transactions from the beginning of time. As a consequence, our applications are not CRUD; they are just CR; nothing ever gets deleted or updated. (Of course, we can take shortcuts, and e.g. store a snapshot of the state every day.)

In conclusion, we have seen what not to do (as prescribed by the three paradigms).

### Part III: Design Principles

#### SRP: The Single Responsibility Principle

_Each software module has one, and only one, reason to change._

Software systems are changed to satisfy stakeholders, i.e. actors. Cohesion is the force that binds together the code responsible to a single actor. Coupling can cause the actions of one actor to affect the dependencies of another. By the SRP, code that different actors depend on must be segregated.

A common solution to this problem is the Facade pattern.

At the level of components, this principle is called the Common Closure Principle.

#### OCP: The Open/Closed Principle

_For software systems to be easy to change, they must be designed to allow the behaviour of those systems to be changed by adding new code, rather than changing existing code._

A software artifact outght to be extendible without having to modify that artifact.

This goal is accomplished by partitioning the system into components, and arranging those components into a dependency hierarchy that protects higher-level components from changes in lower-level components.

#### LSP: The Liskov Substitution Principle

_To build software systems from interchangeable parts, those parts must adhere to a contract that allows those parts to be substituted for one another._

A simple violation of substitutability can cause a sytem's architecture to be polluted with a significant amount of extra mechanisms.

#### ISP: The Interface Segregation Principle

_Avoid depending on things that aren't used._

#### DIP: The Dependency Inversion Principle

_High-level policies should not depend on low-level details. Rather, details should depend on policies._

The most flexible systems are those in which source code dependencies refer only to abstractions, not to concretions.

Changes to concrete implementations do not always, or even usually, require changes to the interfaces that they implement. Good software designers and architects work hard to reduce the volatility of interfaces.

- Don't refer to volatile concrete classes. Refer to abstract interfaces instead. Generally, enforce the use of abstract factories.
- Inheritance is the strongest, and most rigid, of all the source code relationships. Consequently, it should be used with great care.

### Part IV: Component Principles

#### Components

Components are the units of deployment. Well-designed components always retain the ability to be independently deployable and, therefore, independently developable.

These dynamically linked files, which can be plugged together at runtime, are the software components of our architectures.

#### Component Cohesion

##### The Reuse/Release Equivalence Principle

The granule of reuse is the granule of release. Without release numbers, there would be no way to ensure that all the reused components are compatible with each other.

##### The Common Closure Principle

Gather into components those classes that change for the same reasons and at the same times. Separate into different components those classes that change at different times and for different reaseons.

##### The Common Reuse Principle

Don't force users of a component to depend on things they don't need.

#### Component Coupling

##### The Stable Dependencies Problem

Depend in the direction of stability. Any component that we expect to be volatile should not be depended on by a component that is difficult to change. Otherwise, the volatile component will also be difficult to change.

##### The Stable Abstractions Principle

A component should be as abstract as it is stable.

### Part V: Architecture

#### What is Architecture?

The goal of the architect is to create a shape for the system that recognises policy as the most essential element of the system while making the details irrelevant to that policy. This allows decisions about those details to be delayed and deferred.

#### Independence

A good architecture will allow a system to be born as a monolith, deployed in a single file, but then to grow into a set of independently deployable units and then all the way to independent services and/or microservices. Later, as things change, it should allow for reversing that progression and sliding all the way down into a monolith.

#### Boundaries: Drawing Lines

Which decisions are premature? Decisions that have nothing to do with the business requirements.

You draw lines between things that matter and that don't. The GUI doesn't matter to the business rules, so you draw a line between them. Then you arrange the code in those components such that the arrows between them point in one direction - toward the core business. This is an application of the Dependency Inversion and Stable Abstractions principles, where dependency arrows are arranged to point from lower-level details to higher-level abstractions.

#### Boundary Anatomy

The strongest boundary is a service. Services do not depend on their physical location. Communications at this level must deal with high levels of latency.

#### Policy and Level

The farther a policy is from both the inputs and the outputs of the system, the higher its level. We want source code dependencies to be decoupled from data flow and coupled to level, with all source code dependencies pointing in the direction of the higher-level policies, and thus reducing the impact of change.

#### Business Rules

Business rules are rules that make or save the business money.

An Entity is an object within our computer system that embodies a small set of critical business rules operating on Critical Business Data. It is unsullied with concerns about databases, user interfaces, or third-party frameworks. _The Entity is pure business and nothing else._

_Use cases contain the rules that specify how and when the Critical Business Rules within the Entities are invoked._ Use cases control the dance of the Entities. Entities have no knowledge of the use cases that control them. The use case class accepts simple request data structures for its input, and returns simple response data structures as its output, that are not dependent on anything - they could be used by web or any other interfaces.

#### Screaming Architecture

Good architectures are centred on use cases so that architectures can safely describe the structures that support those use cases without committing to frameworks, tools, and environments. The fact that your application is delivered over the web is a detail and should not dominate your system structure.

_If your system architecture is all about the use cases, and if you have kept your frameworks at arm's length, then you should be able to unit-test all those use cases without any of the frameworks in place._ You shouldn't need the web server running to run your tests. You shouldn't need the database connected to run your tests. Your Entity objects should be plain old objects that have no dependencies on frameworks or databases or other complications. Your use cases should coordinate your Entity objects.

#### The Clean Architecture

Layers

- independent of frameworks
- independent of the UI
- independent of the database
- independent of any external agency
- testable - the business rules can be tested without the UI, database, web server, or any other external environment

![The Clean Architecture](https://blog.cleancoder.com/uncle-bob/images/2012-08-13-the-clean-architecture/CleanArchitecture.jpg)

##### The Dependency Rule

The overriding rule that makes this architecture work is the Dependency Rule: _Source code dependencies must point only inward, toward higher-level policies._

Nothing in an inner circle can know anything at all about something in an outer circle. The name of something declared in an outer circle must not be mentioned in an inner circle. Data formats declared in an outer circle (e.g. http requests and responses) should not be used in an inner circle.

##### Crossing Boundary

The Dependency Inversion Principle allows for crossing boundaries, e.g. from controllers through use cases to presenters - through the use of interfaces.

#### Presenters and Humble Objects

_The Humble Object pattern is used to divide something that is hard to test from something that is easy to test._

The View is the humble object that is hard to test. The code in this object is kept as simple as possible. It moves data into the GUI but does not process that data.

The Presenter is the testable object. Its job is to accept data from the application and format it for presentation so that the View can simply move it to the screen. This is the View Model; thus there is nothing left for the View to do other than to load the data from the View Model into the screen, it is humble.

The Database gateways are polymorphic interfaces that contain methods for every create, read, update, or delete operation that can be performed by the application on the database. The interactors are testable, because the gateways can be replaced with appropriate stubs and test doubles. ORMs form another kind of Humble Object boundary between gateway interfaces and the database.

#### Partial Boundaries

The Facade or Strategy pattern can be used here.

#### Layers and Boundaries

#### The Main Component

Think of `main` as a plugin to the application - a plugin that sets up the initial conditions and configurations, gathers all the outside resources, and then hands control over to the high-level policy of the application. Since it is a plugin, it is possible to have many `main` components, one for each configuration of your application.

#### Services: Great and Small

To deal with the cross-cutting concerns that all significant systems face, services must be designed with internal component architectures that follow the Dependency Rule. Those services do not define the architectural boundaries of the system; instead, the components within the services do.

#### The Test Boundary

Tests, by their very nature, follow the Dependency Rule; they are very detailed and concrete; and they always depend inward toward the code being tested. In fact, you can think of the tests as the outermost circle in the architecture. Nothing within the system depends on the tests, and the tests always depend inward on the components of the system.

#### Clean Embedded Architecture

### Part VI: Details

#### The Database Is a Detail

#### The Web Is a Detail

#### Frameworks Are Details

#### The Missing Chapter

##### Package by Layer

##### Package by Feature

##### Ports and Adapters

##### Package by Component

##### The Devil Is in the Implementation Details

##### Organization versus Encapsulation

##### Other Decoupling Modes

##### Conclusion

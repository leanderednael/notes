# CS

## Asymptotic Runtime

Amortized Runtime:

Suppose you have an empty array of size $n$, and wish to insert values into it. The first $n$ inserts take $O(1)$ time each. On insert $n+1$, the array is full and must be resized, which requires copying all of the values over to a new, larger array, in time $O(n)$.

Any given insert operation takes as much as $O(n)$ time, but this is only the case when the array must be resized; otherwise, inserting takes constant time. What is the average time required for each insert operation?

If we increase the array size by a constant amount each time, then the number of elements that eventually need to be copied over will dominate the constant, and operations will take, on average, $O(n)$ time.

However, if we double the size of the array each time, then the total time to add $n$ items is $n+n$, and each insert takes an average of $2n/n = O(1)$ time.

Often we don't actually care how long any given operation takes; what's important is the total time required over all operations. When each expensive operation can be paired with many cheap operations, we can amortize the cost of the expensive operation over all of the cheap ones. Thus, even though any given operation may take $O(n)$ time, the amortized cost may be much lower. In the array example, simply looking at the worst-case runtime for a given step would lead us to conclude that a series of $n$ inserts would be $O(n^2)$, but the amortized analysis shows that (provided we double the number of elements when resizing the array) it will actually be only $O(n)$.

## Computer Arithmetic

Why use bit shifting when it's logically equivalent to multiplying or dividing by two? Because it's fast. Each left shift by one position multiplies the number by 2. Each right shift by one position divides the number by 2.

Binary operators are often used for masking. Suppose an object has a number of Boolean properties. We can combine them into one variable by assigning each to be a power of two and ORing them together: $2^0$ = `1 << 0` = 1, $2^1$ = `1 << 1` = 2, $2^2$ = `1 << 2` = 4, $2^3$ = `1 << 3` = 8. Then if an object has properties one, two, and four, the flag variable will be `0001 | 0010 | 1000 = 1011`. Whether the object has properties three and four can be checked with `(0100 & 1011) = 0` and `(1000 & 1011) = 1`. Binary NOT can be used to flip a property: `~1000 = 0111` and thus `0111 & 1011 = 0011` sets the fourth property to false.

## Data Structures

A heap satisfies the heap ordering property, either min-heap (the value of each node is no smaller than the value of its parent) or max-heap (the value of each node is no larger than the value of its parent).

Hash tables when we want direct access to unsorted data based on a key and there exists a fast-running function for generating the key for each object (assuming the object itself isn't its own key); not when want to sort, or objects are not well distributed (that is, many elements hash to few locations), or when a common use case is to access blocks of sequential items.

Sets and posets.

## Graphs

Adjacency lists and matrices.

### Data Structures on Graphs

A binary search tree (BST) is a rooted binary tree defined recursively: the key of the root is greater than or equal to the key of its left child and less than or equal to the key of its left child and less than or equal to the key of its right child (if any), and this is also true for the subtree rooted at every other node. Operations on a binary search tree take time proportional to the height of the tree, which is the length of the longest chain from the root (which has height zero) to a leaf. This is $\Theta(\log{n})$ in the average case, but $O(n)$ in the worst case.

A self-balancing binary search tree is a BST that automatically keeps its height small (compared to the number of levels required) regardless of any additions or deletions. Self-balancing trees include red-black trees, splay trees, and treaps.

Heaps.

### Graph Algorithms

BFS, DFS, Dijkstra

## Sorting Algorithms

For small sets: bubble, insertion sort.

For large sets: Heapsort, Mergesort, Quicksort.

## Algorithms

Brute Force.

Dynamic Programming and (Greedy Algorithms) are useful for problems which exhibit optimal substructure.

## Complexity Theory

P, NP-hard, NP-complete.

## Languages and State Machines

## Turing Machines

## Security

Confidentiality.

Integrity.

Availability.

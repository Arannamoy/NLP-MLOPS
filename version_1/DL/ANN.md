```mermaid
graph LR;
    A((ANN)) --> B((Classification))
    A((ANN)) --> C((Regression))
    B((Classification)) --> D((Tabular Data))
    C((Regression)) --> D((Tabular Data))
```

- Sequence of data does not matter for training.

`ANN`
```mermaid
graph TD;
  A((A)) --> E((E))
  A((A)) --> F((F))
  A((A)) --> G((G))
  A((A)) --> H((H))
  B((B)) --> E((E))
  B((B)) --> F((F))
  B((B)) --> G((G))
  B((B)) --> H((H))
  C((C)) --> E((E))
  C((C)) --> F((F))
  C((C)) --> G((G))
  C((C)) --> H((H))
  D((D)) --> E((E))
  D((D)) --> F((F))
  D((D)) --> G((G))
  D((D)) --> H((H))
  E((E)) --> O((O))
  F((F)) --> O((O))
  G((G)) --> O((O))
  H((H)) --> O((O))
```

`RNN`
```mermaid
graph TD;
  subgraph Input_Layer [Input Layer]
  A[A] --> E[E]
  A[A] --> F[F]
  A[A] --> G[G]
  A[A] --> H[H]
  B[B] --> E[E]
  B[B] --> F[F]
  B[B] --> G[G]
  B[B] --> H[H]
  C[C] --> E[E]
  C[C] --> F[F]
  C[C] --> G[G]
  C[C] --> H[H]
  D[D] --> E[E]
  D[D] --> F[F]
  D[D] --> G[G]
  D[D] --> H[H]
  end
  
  E[E] --> O[O]
  F[F] --> O[O]
  G[G] --> O[O]
  H[H] --> O[O]
  
    E[E] --> E[E]
    E[E] --> F[F]
    E[E] --> G[G]
    E[E] --> H[H]

    F[F] --> E[E]
    F[F] --> F[F]
    F[F] --> G[G]
    F[F] --> H[H]


    G[G] --> E[E]
    G[G] --> F[F]
    G[G] --> G[G]
    G[G] --> H[H]


    H[H] --> E[E]
    H[H] --> F[F]
    H[H] --> G[G]
    H[H] --> H[H]
```
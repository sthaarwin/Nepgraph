# k-Sweep Results Summary

## Optimal k per Window

| Window | Optimal k | Modularity Q | Largest Comm. % | Components | Communities |
|--------|-----------|-------------|-----------------|------------|-------------|
| 1yr    | 3         | 0.6969      | 19.4%           | 1          | 8           |
| 3yr    | 3         | 0.7459      | 22.9%           | 1          | 8           |
| 5yr    | 3         | 0.7439      | 35.8%           | 1          | 7           |
| max    | 5         | 0.6579      | 27.4%           | 1          | 7           |

## Selection: k=5

k=5 is the smallest value that produces a single connected component
across all time windows. k=3 fragments the max window into 3
disconnected components. At k=5, all windows achieve Q >= 0.66 with
5-7 communities and no community exceeding 37% of nodes.

## Selection Criteria

1. Modularity Q within 1% of maximum (plateau detection)
2. Largest community < 60% of nodes
3. Single connected component
4. 3-20 communities
5. Tiebreak: smaller k (parsimony)
